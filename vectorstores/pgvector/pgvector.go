package pgvector

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/pgvector/pgvector-go"
)

const (
	// pgLockIDEmbeddingTable is used for advisor lock to fix issue arising from concurrent
	// creation of the embedding table.The same value represents the same lock.
	pgLockIDEmbeddingTable = 1573678846307946494
	// pgLockIDCollectionTable is used for advisor lock to fix issue arising from concurrent
	// creation of the collection table.The same value represents the same lock.
	pgLockIDCollectionTable = 1573678846307946495
	// pgLockIDExtension is used for advisor lock to fix issue arising from concurrent creation
	// of the vector extension. The value is deliberately set to the same as python langchain
	// https://github.com/langchain-ai/langchain/blob/v0.0.340/libs/langchain/langchain/vectorstores/pgvector.py#L167
	pgLockIDExtension = 1573678846307946496
)

var (
	ErrEmbedderWrongNumberVectors = errors.New("number of vectors from embedder does not match number of documents")
	ErrInvalidScoreThreshold      = errors.New("score threshold must be between 0 and 1")
	ErrInvalidFilters             = errors.New("invalid filters")
	ErrInvalidFilterKey           = errors.New("filter key must be a bare identifier")
	ErrInvalidMetadataIndex       = errors.New("invalid metadata index")
	ErrUnsupportedOptions         = errors.New("unsupported options")
)

// PGXConn represents both a pgx.Conn and pgxpool.Pool conn.
type PGXConn interface {
	Ping(ctx context.Context) error
	Begin(ctx context.Context) (pgx.Tx, error)
	Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error)
	Query(ctx context.Context, sql string, arguments ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, arguments ...any) pgx.Row
	SendBatch(ctx context.Context, batch *pgx.Batch) pgx.BatchResults
}

type CloseNoErr interface {
	Close()
}

// Store is a wrapper around the pgvector client.
type Store struct {
	embedder            embeddings.Embedder
	connURL             string
	conn                PGXConn
	embeddingTableName  string
	collectionTableName string
	collectionName      string
	collectionUUID      string
	collectionMetadata  map[string]any
	preDeleteCollection bool
	vectorDimensions    int
	hnswIndex           *HNSWIndex
	metadataIndexes     []MetadataIndex
}

type HNSWIndex struct {
	m                int
	efConstruction   int
	distanceFunction string
}

var _ vectorstores.VectorStore = Store{}

// New creates a new Store with options.
func New(ctx context.Context, opts ...Option) (Store, error) {
	store, err := applyClientOptions(opts...)
	if err != nil {
		return Store{}, err
	}
	if store.conn == nil {
		store.conn, err = pgx.Connect(ctx, store.connURL)
		if err != nil {
			return Store{}, err
		}
	}
	if err = store.conn.Ping(ctx); err != nil {
		return Store{}, err
	}
	if err = store.init(ctx); err != nil {
		return Store{}, err
	}
	return store, nil
}

// Close closes the connection.
func (s Store) Close() error {
	if closer, ok := s.conn.(io.Closer); ok {
		return closer.Close()
	}
	if closer, ok := s.conn.(CloseNoErr); ok {
		closer.Close()
	}
	return nil
}

func (s *Store) init(ctx context.Context) (err error) {
	tx, err := s.conn.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback(context.WithoutCancel(ctx))
		}
	}()

	if err := s.createVectorExtensionIfNotExists(ctx, tx); err != nil {
		return err
	}
	if err := s.createCollectionTableIfNotExists(ctx, tx); err != nil {
		return err
	}
	if err := s.createEmbeddingTableIfNotExists(ctx, tx); err != nil {
		return err
	}
	if err := s.createMetadataIndexesIfNotExist(ctx, tx); err != nil {
		return err
	}
	if s.preDeleteCollection {
		if err := s.RemoveCollection(ctx, tx); err != nil {
			return err
		}
	}
	if err := s.createOrGetCollection(ctx, tx); err != nil {
		return err
	}

	return tx.Commit(ctx)
}

func (s Store) createVectorExtensionIfNotExists(ctx context.Context, tx pgx.Tx) error {
	// inspired by
	// https://github.com/langchain-ai/langchain/blob/v0.0.340/libs/langchain/langchain/vectorstores/pgvector.py#L167
	// The advisor lock fixes issue arising from concurrent
	// creation of the vector extension.
	// https://github.com/langchain-ai/langchain/issues/12933
	// For more information see:
	// https://www.postgresql.org/docs/16/explicit-locking.html#ADVISORY-LOCKS
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", pgLockIDExtension); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector"); err != nil {
		return err
	}
	return nil
}

func (s Store) createCollectionTableIfNotExists(ctx context.Context, tx pgx.Tx) error {
	// inspired by
	// https://github.com/langchain-ai/langchain/blob/v0.0.340/libs/langchain/langchain/vectorstores/pgvector.py#L167
	// The advisor lock fixes issue arising from concurrent
	// creation of the vector extension.
	// https://github.com/langchain-ai/langchain/issues/12933
	// For more information see:
	// https://www.postgresql.org/docs/16/explicit-locking.html#ADVISORY-LOCKS
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", pgLockIDCollectionTable); err != nil {
		return err
	}
	sql := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
	name varchar,
	cmetadata json,
	"uuid" uuid NOT NULL,
	UNIQUE (name),
	PRIMARY KEY (uuid))`, s.collectionTableName)
	if _, err := tx.Exec(ctx, sql); err != nil {
		return err
	}
	return nil
}

func (s Store) createEmbeddingTableIfNotExists(ctx context.Context, tx pgx.Tx) error {
	// inspired by
	// https://github.com/langchain-ai/langchain/blob/v0.0.340/libs/langchain/langchain/vectorstores/pgvector.py#L167
	// The advisor lock fixes issue arising from concurrent
	// creation of the vector extension.
	// https://github.com/langchain-ai/langchain/issues/12933
	// For more information see:
	// https://www.postgresql.org/docs/16/explicit-locking.html#ADVISORY-LOCKS
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", pgLockIDEmbeddingTable); err != nil {
		return err
	}

	vectorDimensions := ""
	if s.vectorDimensions > 0 {
		vectorDimensions = fmt.Sprintf("(%d)", s.vectorDimensions)
	}

	sql := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
	collection_id uuid,
	embedding vector%s,
	document varchar,
	cmetadata json,
	"uuid" uuid NOT NULL,
	CONSTRAINT langchain_pg_embedding_collection_id_fkey
	FOREIGN KEY (collection_id) REFERENCES %s (uuid) ON DELETE CASCADE,
	PRIMARY KEY (uuid))`, s.embeddingTableName, vectorDimensions, s.collectionTableName)
	if _, err := tx.Exec(ctx, sql); err != nil {
		return err
	}
	sql = fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_collection_id ON %s (collection_id)`, s.embeddingTableName, s.embeddingTableName)
	if _, err := tx.Exec(ctx, sql); err != nil {
		return err
	}

	// See this for more details on HNWS indexes: https://github.com/pgvector/pgvector#hnsw
	if s.hnswIndex != nil {
		sql = fmt.Sprintf(
			`CREATE INDEX IF NOT EXISTS %s_embedding_hnsw ON %s USING hnsw (embedding %s)`,
			s.embeddingTableName, s.embeddingTableName, s.hnswIndex.distanceFunction,
		)
		if s.hnswIndex.m > 0 && s.hnswIndex.efConstruction > 0 {
			sql = fmt.Sprintf("%s WITH (m=%d, ef_construction = %d)", sql, s.hnswIndex.m, s.hnswIndex.efConstruction)
		}
		if _, err := tx.Exec(ctx, sql); err != nil {
			return err
		}
	}

	return nil
}

// AddDocuments adds documents to the Postgres collection associated with 'Store'.
// and returns the ids of the added documents.
func (s Store) AddDocuments(
	ctx context.Context,
	docs []schema.Document,
	options ...vectorstores.Option,
) ([]string, error) {
	opts := s.getOptions(options...)
	if opts.ScoreThreshold != 0 || opts.Filters != nil || opts.NameSpace != "" {
		return nil, ErrUnsupportedOptions
	}

	docs = s.deduplicate(ctx, opts, docs)

	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		texts = append(texts, doc.PageContent)
	}

	embedder := s.embedder
	if opts.Embedder != nil {
		embedder = opts.Embedder
	}
	vectors, err := embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return nil, err
	}

	if len(vectors) != len(docs) {
		return nil, ErrEmbedderWrongNumberVectors
	}

	b := &pgx.Batch{}
	sql := fmt.Sprintf(`INSERT INTO %s (uuid, document, embedding, cmetadata, collection_id)
		VALUES($1, $2, $3, $4, $5)`, s.embeddingTableName)

	ids := make([]string, len(docs))
	for docIdx, doc := range docs {
		id := uuid.New().String()
		ids[docIdx] = id
		b.Queue(sql, id, doc.PageContent, pgvector.NewVector(vectors[docIdx]), doc.Metadata, s.collectionUUID)
	}
	return ids, s.conn.SendBatch(ctx, b).Close()
}

//nolint:cyclop
func (s Store) SimilaritySearch(
	ctx context.Context,
	query string,
	numDocuments int,
	options ...vectorstores.Option,
) ([]schema.Document, error) {
	opts := s.getOptions(options...)
	collectionName := s.getNameSpace(opts)
	scoreThreshold, err := s.getScoreThreshold(opts)
	if err != nil {
		return nil, err
	}
	filter, err := s.getFilters(opts)
	if err != nil {
		return nil, err
	}
	embedder := s.embedder
	if opts.Embedder != nil {
		embedder = opts.Embedder
	}
	embedderData, err := embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	args := make([]any, 0, 5+len(filter))
	args = append(args, len(embedderData), pgvector.NewVector(embedderData), numDocuments, collectionName)

	// Every predicate that can reject a row belongs inside the fence, so that a
	// distance is computed only for the rows that survive it, and so that a
	// metadata index is reachable at the scan. Hoisting them out, as this query
	// once did, costs a full table scan and a detoast of every stored vector.
	innerQuerys, filterArgs, err := filterPredicates(s.embeddingTableName+".", filter, len(args))
	if err != nil {
		return nil, err
	}
	args = append(args, filterArgs...)
	innerQuery := strings.Join(innerQuerys, " AND ")
	if len(innerQuery) == 0 {
		innerQuery = "TRUE"
	}

	// The threshold is the one predicate that cannot move inside: it reads the
	// distance the fenced select computes.
	outerQuery := "TRUE"
	if scoreThreshold != 0 {
		args = append(args, float64(1-scoreThreshold))
		outerQuery = fmt.Sprintf("data.distance < $%d", len(args))
	}

	// AS MATERIALIZED is load-bearing rather than decorative. It is what orders
	// vector_dims() ahead of <=>, and a table holding more than one embedding
	// dimension -- any table written before an embedding model was changed --
	// fails outright with "different vector dimensions" without it.
	sql := fmt.Sprintf(`WITH filtered_embedding_dims AS MATERIALIZED (
	SELECT
		%[1]s.uuid,
		%[1]s.document,
		%[1]s.cmetadata,
		%[1]s.embedding <=> $2 AS distance
	FROM
		%[1]s
	WHERE %[1]s.collection_id = (SELECT %[2]s.uuid FROM %[2]s WHERE %[2]s.name = $4 ORDER BY %[2]s.name LIMIT 1)
		AND vector_dims(%[1]s.embedding) = $1
		AND %[3]s
)
SELECT
	data.document,
	data.cmetadata,
	(1 - data.distance) AS score
FROM filtered_embedding_dims AS data
WHERE %[4]s
ORDER BY
	data.distance,
	data.uuid
LIMIT $3`, s.embeddingTableName, s.collectionTableName, innerQuery, outerQuery)
	rows, err := s.conn.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	docs := make([]schema.Document, 0)
	for rows.Next() {
		doc := schema.Document{}
		if err := rows.Scan(&doc.PageContent, &doc.Metadata, &doc.Score); err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}
	return docs, rows.Err()
}

//nolint:cyclop
func (s Store) Search(
	ctx context.Context,
	numDocuments int,
	options ...vectorstores.Option,
) ([]schema.Document, error) {
	opts := s.getOptions(options...)
	collectionName := s.getNameSpace(opts)
	filter, err := s.getFilters(opts)
	if err != nil {
		return nil, err
	}
	args := make([]any, 0, 2+len(filter))
	args = append(args, numDocuments, collectionName)
	whereQuerys, filterArgs, err := filterPredicates(s.embeddingTableName+".", filter, len(args))
	if err != nil {
		return nil, err
	}
	args = append(args, filterArgs...)
	whereQuery := strings.Join(whereQuerys, " AND ")
	if len(whereQuery) == 0 {
		whereQuery = "TRUE"
	}
	sql := fmt.Sprintf(`SELECT
	%s.document,
	%s.cmetadata
FROM %s
JOIN %s ON %s.collection_id=%s.uuid
WHERE %s.name=$2 AND %s
LIMIT $1`, s.embeddingTableName, s.embeddingTableName, s.embeddingTableName,
		s.collectionTableName, s.embeddingTableName, s.collectionTableName, s.collectionTableName,
		whereQuery)
	rows, err := s.conn.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	docs := make([]schema.Document, 0)
	defer rows.Close()

	for rows.Next() {
		doc := schema.Document{}
		if err := rows.Scan(&doc.PageContent, &doc.Metadata); err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}
	return docs, rows.Err()
}

func (s Store) DropTables(ctx context.Context) error {
	if _, err := s.conn.Exec(ctx, fmt.Sprintf(`DROP TABLE IF EXISTS %s`, s.embeddingTableName)); err != nil {
		return err
	}
	if _, err := s.conn.Exec(ctx, fmt.Sprintf(`DROP TABLE IF EXISTS %s`, s.collectionTableName)); err != nil {
		return err
	}
	return nil
}

func (s Store) RemoveCollection(ctx context.Context, tx pgx.Tx) error {
	_, err := tx.Exec(ctx, fmt.Sprintf(`DELETE FROM %s WHERE name = $1`, s.collectionTableName), s.collectionName)
	return err
}

func (s *Store) createOrGetCollection(ctx context.Context, tx pgx.Tx) error {
	sql := fmt.Sprintf(`INSERT INTO %s (uuid, name, cmetadata)
		VALUES($1, $2, $3) ON CONFLICT (name) DO
		UPDATE SET cmetadata = $3`, s.collectionTableName)
	if _, err := tx.Exec(ctx, sql, uuid.New().String(), s.collectionName, s.collectionMetadata); err != nil {
		return err
	}
	sql = fmt.Sprintf(`SELECT uuid FROM %s WHERE name = $1 ORDER BY name limit 1`, s.collectionTableName)
	return tx.QueryRow(ctx, sql, s.collectionName).Scan(&s.collectionUUID)
}

// getOptions applies given options to default Options and returns it
// This uses options pattern so clients can easily pass options without changing function signature.
func (s Store) getOptions(options ...vectorstores.Option) vectorstores.Options {
	opts := vectorstores.Options{}
	for _, opt := range options {
		opt(&opts)
	}
	return opts
}

func (s Store) getNameSpace(opts vectorstores.Options) string {
	if opts.NameSpace != "" {
		return opts.NameSpace
	}
	return s.collectionName
}

func (s Store) getScoreThreshold(opts vectorstores.Options) (float32, error) {
	if opts.ScoreThreshold < 0 || opts.ScoreThreshold > 1 {
		return 0, ErrInvalidScoreThreshold
	}
	return opts.ScoreThreshold, nil
}

// filterPredicates renders the metadata filter as predicates numbered from
// argOffset; the returned args must be appended to the query in the same order.
// Keys are sorted so that one filter always renders as one statement text.
//
// The key is inlined and only the value is bound, because `cmetadata ->> $n` is
// not the expression a metadata index was built over. A key that fails the gate
// is an error rather than a dropped predicate: a caller's filter may be the only
// thing keeping one tenant's documents out of another's results.
func filterPredicates(prefix string, filter map[string]any, argOffset int) ([]string, []any, error) {
	keys := make([]string, 0, len(filter))
	for k := range filter {
		if !metadataKeyPattern.MatchString(k) {
			return nil, nil, fmt.Errorf("%w: %q", ErrInvalidFilterKey, k)
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)

	predicates := make([]string, 0, len(keys))
	args := make([]any, 0, len(keys))
	for _, k := range keys {
		predicates = append(predicates, fmt.Sprintf("(%scmetadata ->> '%s') = $%d",
			prefix, k, argOffset+len(args)+1))
		args = append(args, fmt.Sprintf("%v", filter[k]))
	}
	return predicates, args, nil
}

// getFilters return metadata filters, now only support map[key]value pattern
// TODO: should support more types like {"key1": {"key2":"values2"}} or {"key": ["value1", "values2"]}.
func (s Store) getFilters(opts vectorstores.Options) (map[string]any, error) {
	if opts.Filters != nil {
		if filters, ok := opts.Filters.(map[string]any); ok {
			return filters, nil
		}
		return nil, ErrInvalidFilters
	}
	return map[string]any{}, nil
}

func (s Store) deduplicate(
	ctx context.Context,
	opts vectorstores.Options,
	docs []schema.Document,
) []schema.Document {
	if opts.Deduplicater == nil {
		return docs
	}

	filtered := make([]schema.Document, 0, len(docs))
	for _, doc := range docs {
		if !opts.Deduplicater(ctx, doc) {
			filtered = append(filtered, doc)
		}
	}

	return filtered
}
