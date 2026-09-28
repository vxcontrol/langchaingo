package pgvector

import (
	"context"
	"fmt"
	"hash/fnv"
	"maps"
	"regexp"
	"slices"
	"strings"

	"github.com/jackc/pgx/v5"
)

// metadataKeyPattern gates every metadata key that reaches the statement text.
// A key that travels as a bind parameter makes `cmetadata ->> $n` a different
// expression from the `cmetadata ->> 'key'` an index was built over, so the
// index becomes unreachable as soon as the planner chooses a generic plan.
// Inlining the key is what keeps the index in play, and nothing this pattern
// admits needs quoting.
var metadataKeyPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]{0,62}$`)

// maxIdentifierLen is PostgreSQL's NAMEDATALEN-1: a longer name is silently
// truncated, which would collide two distinct declarations under one index.
const maxIdentifierLen = 63

// MetadataIndex declares an index over keys inside the cmetadata column. The
// store creates every declared index it does not already find when it starts.
//
// Declaring an index is only half the work: a query reaches it solely when it
// constrains Keys from the first one onwards, so the leading key belongs to
// whichever filter every caller of that search path supplies.
type MetadataIndex struct {
	// Name overrides the derived name. Leave it empty unless an existing index
	// has to be adopted under the name it already carries.
	Name string

	// Keys are the cmetadata keys in index order, most selective predicate
	// first. At least one is required.
	Keys []string

	// Exclude restricts the index to rows whose key does NOT hold the given
	// value. It earns its place when the interesting rows are a small minority
	// of the table: the index then spans that minority rather than every row.
	Exclude map[string]string
}

func (m MetadataIndex) validate() error {
	if len(m.Keys) == 0 {
		return fmt.Errorf("%w: an index needs at least one key", ErrInvalidMetadataIndex)
	}
	for _, key := range m.Keys {
		if !metadataKeyPattern.MatchString(key) {
			return fmt.Errorf("%w: key %q is not a bare identifier", ErrInvalidMetadataIndex, key)
		}
	}
	for key := range m.Exclude {
		if !metadataKeyPattern.MatchString(key) {
			return fmt.Errorf("%w: excluded key %q is not a bare identifier", ErrInvalidMetadataIndex, key)
		}
	}
	if m.Name != "" && !metadataKeyPattern.MatchString(m.Name) {
		return fmt.Errorf("%w: name %q is not a bare identifier", ErrInvalidMetadataIndex, m.Name)
	}
	return nil
}

func (m MetadataIndex) indexName(table string) string {
	if m.Name != "" {
		return m.Name
	}

	parts := make([]string, 0, len(m.Keys)+3)
	parts = append(parts, strings.ReplaceAll(table, ".", "_"), "meta")
	parts = append(parts, m.Keys...)
	if len(m.Exclude) > 0 {
		parts = append(parts, "partial")
	}
	name := strings.Join(parts, "_")
	prefix := strings.ToValidUTF8(name[:min(len(name), maxIdentifierLen-9)], "")
	return fmt.Sprintf("%s_%08x", prefix, m.fingerprint(table))
}

func (m MetadataIndex) fingerprint(table string) uint32 {
	sum := fnv.New32a()
	_, _ = sum.Write([]byte(m.definition(table)))
	return sum.Sum32()
}

func (m MetadataIndex) definition(table string) string {
	var b strings.Builder
	_, _ = fmt.Fprintf(&b, "%q", table)
	for _, key := range m.Keys {
		_, _ = fmt.Fprintf(&b, " k%q", key)
	}
	for _, key := range m.excludedKeys() {
		_, _ = fmt.Fprintf(&b, " x%q=%q", key, m.Exclude[key])
	}
	return b.String()
}

func (m MetadataIndex) excludedKeys() []string {
	return slices.Sorted(maps.Keys(m.Exclude))
}

// ddl renders the CREATE INDEX for this declaration. Index expressions and
// predicates admit no bind parameters, so every part is rendered as literal
// text; validate must pass first.
func (m MetadataIndex) ddl(table string) (string, error) {
	if err := m.validate(); err != nil {
		return "", err
	}

	columns := make([]string, 0, len(m.Keys))
	for _, key := range m.Keys {
		columns = append(columns, fmt.Sprintf("(cmetadata ->> '%s')", key))
	}

	statement := fmt.Sprintf("CREATE INDEX IF NOT EXISTS %s ON %s (%s)",
		m.indexName(table), table, strings.Join(columns, ", "))

	if len(m.Exclude) == 0 {
		return statement, nil
	}

	keys := m.excludedKeys()
	predicates := make([]string, 0, len(keys))
	for _, key := range keys {
		literal, err := quoteLiteral(m.Exclude[key])
		if err != nil {
			return "", err
		}
		// IS DISTINCT FROM rather than <>, so a row without the key is indexed
		// instead of being dropped by a NULL comparison.
		predicates = append(predicates,
			fmt.Sprintf("(cmetadata ->> '%s') IS DISTINCT FROM %s", key, literal))
	}

	return statement + " WHERE " + strings.Join(predicates, " AND "), nil
}

// quoteLiteral renders a string as an SQL literal. Doubling the quote is the
// whole of the escaping under standard_conforming_strings, which has been on by
// default since PostgreSQL 9.1; a backslash is rejected rather than trusted to
// it, and a NUL byte cannot appear in a statement at all.
func quoteLiteral(value string) (string, error) {
	if strings.ContainsAny(value, "\x00\\") {
		return "", fmt.Errorf("%w: value %q may not contain a backslash or a NUL byte",
			ErrInvalidMetadataIndex, value)
	}
	return "'" + strings.ReplaceAll(value, "'", "''") + "'", nil
}

// createMetadataIndexesIfNotExist brings the declared indexes into existence.
//
// It runs inside init's transaction, which already holds the embedding table's
// advisory lock, so concurrent stores opening against one table serialise here
// rather than racing. That also rules out CREATE INDEX CONCURRENTLY, which
// cannot run inside a transaction: the first build takes a ShareLock and blocks
// writers for its duration, and every start afterwards is a catalog lookup.
func (s Store) createMetadataIndexesIfNotExist(ctx context.Context, tx pgx.Tx) error {
	if len(s.metadataIndexes) == 0 {
		return nil
	}

	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", pgLockIDEmbeddingTable); err != nil {
		return err
	}

	definitions := make(map[string]string, len(s.metadataIndexes))
	for _, index := range s.metadataIndexes {
		statement, err := index.ddl(s.embeddingTableName)
		if err != nil {
			return err
		}
		name := index.indexName(s.embeddingTableName)
		folded := strings.ToLower(name)
		definition := index.definition(s.embeddingTableName)
		if declared, ok := definitions[folded]; ok && declared != definition {
			return fmt.Errorf("%w: two declarations share the index name %s", ErrInvalidMetadataIndex, name)
		}
		definitions[folded] = definition
		if _, err := tx.Exec(ctx, statement); err != nil {
			return fmt.Errorf("create metadata index %s: %w", name, err)
		}
	}

	// An expression index carries no statistics of its own until the table is
	// analysed, and until then the planner costs it off the column's own
	// distribution and can reject it outright.
	if _, err := tx.Exec(ctx, "ANALYZE "+s.embeddingTableName); err != nil {
		return err
	}

	return nil
}
