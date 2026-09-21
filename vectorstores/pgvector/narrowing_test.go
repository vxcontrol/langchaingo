package pgvector

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/stretchr/testify/require"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
)

// fixedEmbedder answers with a deterministic unit vector of a fixed width, so a
// search asserts about the statement the store builds rather than about a model.
type fixedEmbedder struct{ dims int }

func (e fixedEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		out[i] = e.vector(text)
	}
	return out, nil
}

func (e fixedEmbedder) EmbedQuery(_ context.Context, text string) ([]float32, error) {
	return e.vector(text), nil
}

func (e fixedEmbedder) vector(text string) []float32 {
	var seed uint32 = 2166136261
	for _, b := range []byte(text) {
		seed ^= uint32(b)
		seed *= 16777619
	}
	vec := make([]float32, e.dims)
	for i := range vec {
		seed = seed*1664525 + 1013904223
		vec[i] = float32(seed%1000)/1000 + 0.001
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] = float32(float64(vec[i]) / norm)
	}
	return vec
}

// These cases live inside the package because they assert about the statement
// the store builds, so they cannot borrow the external suite's helpers.
func narrowingURL(t *testing.T) string {
	t.Helper()

	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	url := os.Getenv("PGVECTOR_CONNECTION_STRING")
	if url == "" {
		t.Skip("PGVECTOR_CONNECTION_STRING is not set")
	}
	return url
}

func narrowingCollection() string {
	return "narrowing-" + uuid.New().String()
}

func newNarrowingStore(t *testing.T, url, collection string, dims int, indexes ...MetadataIndex) Store {
	t.Helper()

	opts := []Option{
		Option(WithConnectionURL(url)),
		WithEmbedder(fixedEmbedder{dims: dims}),
		WithCollectionName(collection),
	}
	if len(indexes) > 0 {
		opts = append(opts, WithMetadataIndexes(indexes...))
	}

	store, err := New(t.Context(), opts...)
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	return store
}

// A table written before an embedding model changed holds more than one vector
// width. The dimension guard has to be evaluated before the distance operator,
// or every search on such a table fails outright.
func TestSimilaritySearchToleratesAMixedWidthTableUnderFiltersAndThreshold(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	collection := narrowingCollection()
	ctx := t.Context()

	wide := newNarrowingStore(t, url, collection, 64)
	_, err := wide.AddDocuments(ctx, []schema.Document{
		{PageContent: "the lamp beside the desk is blue", Metadata: map[string]any{"flow_id": "7", "doc_type": "memory"}},
		{PageContent: "the chair beside the desk is beige", Metadata: map[string]any{"flow_id": "7", "doc_type": "memory"}},
		{PageContent: "the desk is orange", Metadata: map[string]any{"flow_id": "9", "doc_type": "memory"}},
	})
	require.NoError(t, err)

	// The same collection, now written by a narrower model, and landing in the
	// very flow the search below scopes to.
	narrow := newNarrowingStore(t, url, collection, 32)
	_, err = narrow.AddDocuments(ctx, []schema.Document{
		{PageContent: "written after the model changed", Metadata: map[string]any{"flow_id": "7", "doc_type": "memory"}},
	})
	require.NoError(t, err)

	docs, err := wide.SimilaritySearch(ctx, "what colour is the lamp", 3,
		vectorstores.WithScoreThreshold(0.2),
		vectorstores.WithFilters(map[string]any{"flow_id": "7", "doc_type": "memory"}))
	require.NoError(t, err)
	require.NotEmpty(t, docs)
	for _, doc := range docs {
		require.Equal(t, "7", doc.Metadata["flow_id"])
		require.NotEqual(t, "written after the model changed", doc.PageContent)
	}
}

func TestSimilaritySearchKeepsTheFilterScope(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	collection := narrowingCollection()
	ctx := t.Context()

	store := newNarrowingStore(t, url, collection, 64)
	_, err := store.AddDocuments(ctx, []schema.Document{
		{PageContent: "mine", Metadata: map[string]any{"flow_id": "1", "doc_type": "memory"}},
		{PageContent: "theirs", Metadata: map[string]any{"flow_id": "2", "doc_type": "memory"}},
	})
	require.NoError(t, err)

	docs, err := store.SimilaritySearch(ctx, "anything", 10,
		vectorstores.WithFilters(map[string]any{"flow_id": "1"}))
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.Equal(t, "mine", docs[0].PageContent)
}

// A key that cannot be inlined must fail the search rather than widen it: the
// filter it carries may be the only thing scoping the caller's results.
func TestSimilaritySearchRefusesAFilterKeyItCannotInline(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	collection := narrowingCollection()
	ctx := t.Context()

	store := newNarrowingStore(t, url, collection, 64)
	_, err := store.AddDocuments(ctx, []schema.Document{
		{PageContent: "mine", Metadata: map[string]any{"flow_id": "1"}},
		{PageContent: "theirs", Metadata: map[string]any{"flow_id": "2"}},
	})
	require.NoError(t, err)

	_, err = store.SimilaritySearch(ctx, "anything", 10,
		vectorstores.WithFilters(map[string]any{"flow id": "1"}))
	require.ErrorIs(t, err, ErrInvalidFilterKey)
}

func TestStoreCreatesTheDeclaredMetadataIndexes(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()

	declared := []MetadataIndex{
		{Keys: []string{"doc_type", "flow_id"}},
		{Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "memory"}},
	}
	store := newNarrowingStore(t, url, narrowingCollection(), 64, declared...)

	conn, err := pgx.Connect(ctx, url)
	require.NoError(t, err)
	defer conn.Close(ctx)

	for _, index := range declared {
		name := index.indexName(store.embeddingTableName)
		var definition string
		err := conn.QueryRow(ctx,
			"SELECT indexdef FROM pg_indexes WHERE tablename = $1 AND indexname = $2",
			store.embeddingTableName, name).Scan(&definition)
		require.NoErrorf(t, err, "index %s was not created", name)
		for _, key := range index.Keys {
			require.Contains(t, definition, fmt.Sprintf("(cmetadata ->> '%s'::text)", key))
		}
	}

	// A second store against the same table must be a no-op rather than an error.
	newNarrowingStore(t, url, narrowingCollection(), 64, declared...)
}

// The whole point of the rewrite: the scan reads one flow, not the table.
func TestSimilaritySearchReadsOnlyTheFilteredRows(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()

	store := newNarrowingStore(t, url, narrowingCollection(), 64,
		MetadataIndex{Keys: []string{"doc_type", "flow_id"}})

	const flows, perFlow = 200, 25
	docs := make([]schema.Document, 0, flows*perFlow)
	for flow := range flows {
		for i := range perFlow {
			docs = append(docs, schema.Document{
				PageContent: fmt.Sprintf("flow %d document %d", flow, i),
				Metadata:    map[string]any{"flow_id": strconv.Itoa(flow), "doc_type": "memory"},
			})
		}
	}
	for start := 0; start < len(docs); start += 500 {
		_, err := store.AddDocuments(ctx, docs[start:min(start+500, len(docs))])
		require.NoError(t, err)
	}

	conn, err := pgx.Connect(ctx, url)
	require.NoError(t, err)
	defer conn.Close(ctx)
	_, err = conn.Exec(ctx, "ANALYZE "+store.embeddingTableName)
	require.NoError(t, err)

	plan := explainSimilaritySearch(t, ctx, conn, store, "flow 3 document 1",
		map[string]any{"doc_type": "memory", "flow_id": "3"})

	require.Contains(t, plan, store.embeddingTableName+"_meta_doc_type_flow_id",
		"the metadata index must carry the scan:\n%s", plan)
	require.NotContains(t, plan, "Seq Scan on "+store.embeddingTableName,
		"the whole table was read:\n%s", plan)
}

func explainSimilaritySearch(
	t *testing.T, ctx context.Context, conn *pgx.Conn, store Store, query string, filter map[string]any,
) string {
	t.Helper()

	vector, err := store.embedder.EmbedQuery(ctx, query)
	require.NoError(t, err)

	predicates, args, err := filterPredicates(store.embeddingTableName+".", filter, 4)
	require.NoError(t, err)

	literals := make([]string, 0, len(vector))
	for _, v := range vector {
		literals = append(literals, strconv.FormatFloat(float64(v), 'f', -1, 32))
	}

	statement := fmt.Sprintf(`EXPLAIN (COSTS OFF) WITH filtered_embedding_dims AS MATERIALIZED (
	SELECT %[1]s.uuid, %[1]s.document, %[1]s.cmetadata, %[1]s.embedding <=> '[%[2]s]'::vector AS distance
	FROM %[1]s
	WHERE %[1]s.collection_id = (SELECT %[3]s.uuid FROM %[3]s WHERE %[3]s.name = $1 ORDER BY %[3]s.name LIMIT 1)
		AND vector_dims(%[1]s.embedding) = %[4]d
		AND %[5]s
)
SELECT data.document, data.cmetadata, (1 - data.distance) AS score
FROM filtered_embedding_dims AS data
WHERE data.distance < 0.8
ORDER BY data.distance, data.uuid LIMIT 3`,
		store.embeddingTableName, strings.Join(literals, ","), store.collectionTableName,
		len(vector), strings.Join(predicates, " AND "))

	// filterPredicates numbered from 4 to match the live statement; here the
	// collection name is $1 and the filter values follow it.
	statement = strings.NewReplacer("$5", "$2", "$6", "$3", "$7", "$4").Replace(statement)

	params := append([]any{store.collectionName}, args...)
	rows, err := conn.Query(ctx, statement, params...)
	require.NoError(t, err)
	defer rows.Close()

	var plan strings.Builder
	for rows.Next() {
		var line string
		require.NoError(t, rows.Scan(&line))
		plan.WriteString(line)
		plan.WriteString("\n")
	}
	require.NoError(t, rows.Err())
	return plan.String()
}
