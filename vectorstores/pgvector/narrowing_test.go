package pgvector

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	tcpostgres "github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"

	"github.com/vxcontrol/langchaingo/internal/testutil/testctr"
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

var sharedPostgres struct {
	once      sync.Once
	url       string
	err       error
	container *tcpostgres.PostgresContainer
}

// These cases live inside the package because they assert about the statement
// the store builds, so they cannot borrow the external suite's helpers.
func narrowingURL(t *testing.T) string {
	t.Helper()

	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	if url := os.Getenv("PGVECTOR_CONNECTION_STRING"); url != "" {
		return url
	}
	testctr.SkipIfDockerNotAvailable(t)

	sharedPostgres.once.Do(func() {
		ctx := context.Background()
		sharedPostgres.container, sharedPostgres.err = tcpostgres.Run(ctx, "docker.io/pgvector/pgvector:pg16",
			tcpostgres.WithDatabase("db_test"), tcpostgres.WithUsername("user"), tcpostgres.WithPassword("passw0rd!"),
			testcontainers.WithWaitStrategy(wait.ForAll(
				wait.ForLog("database system is ready to accept connections").WithOccurrence(2).
					WithStartupTimeout(60*time.Second),
				wait.ForListeningPort("5432/tcp").WithStartupTimeout(60*time.Second))))
		if sharedPostgres.err == nil {
			sharedPostgres.url, sharedPostgres.err = sharedPostgres.container.ConnectionString(ctx, "sslmode=disable")
		}
	})
	require.NoError(t, sharedPostgres.err)
	return sharedPostgres.url
}

func narrowingCollection() string {
	return "narrowing-" + uuid.New().String()
}

func newNarrowingStore(t *testing.T, url, collection string, dims int) Store {
	t.Helper()

	store, err := New(t.Context(), WithConnectionURL(url), WithEmbedder(fixedEmbedder{dims: dims}),
		WithCollectionName(collection))
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
		vectorstores.WithFilters(map[string]any{"flow\\id": "1"}))
	require.ErrorIs(t, err, ErrInvalidFilterKey)
}

func TestSimilaritySearchFiltersOnAKeyThatIsNotAnIdentifier(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	collection := narrowingCollection()
	ctx := t.Context()

	store := newNarrowingStore(t, url, collection, 64)
	_, err := store.AddDocuments(ctx, []schema.Document{
		{PageContent: "mine", Metadata: map[string]any{"doc-id": "1"}},
		{PageContent: "theirs", Metadata: map[string]any{"doc-id": "2"}},
	})
	require.NoError(t, err)

	docs, err := store.SimilaritySearch(ctx, "anything", 10,
		vectorstores.WithFilters(map[string]any{"doc-id": "1"}))
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.Equal(t, "mine", docs[0].PageContent)
}

func TestStoreCreatesTheDeclaredMetadataIndexes(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()

	declared := []MetadataIndex{
		{Keys: []string{"doc_type", "flow_id"}},
		{Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "memory"}},
		{Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "session"}},
	}
	store, conn := newIsolatedIndexedStore(t, url, declared...)

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
		for _, value := range index.Exclude {
			require.Contains(t, definition, fmt.Sprintf("'%s'::text", value))
		}
	}

	again, err := New(ctx, WithConnectionURL(url), WithEmbedder(fixedEmbedder{dims: 64}),
		WithCollectionName("c"), WithEmbeddingTableName(store.embeddingTableName),
		WithCollectionTableName(store.collectionTableName), WithMetadataIndexes(declared...))
	require.NoError(t, err, "a second store against the same table must be a no-op rather than an error")
	require.NoError(t, again.Close())
}

// The whole point of the rewrite: the scan reads one flow, not the table.
func TestSimilaritySearchReadsOnlyTheFilteredRows(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()

	suffix := strings.ReplaceAll(uuid.NewString(), "-", "")[:12]
	index := MetadataIndex{Keys: []string{"doc_type", "flow_id"}}
	store, err := New(ctx,
		WithConnectionURL(url),
		WithEmbedder(fixedEmbedder{dims: 64}),
		WithCollectionName(narrowingCollection()),
		WithCollectionTableName("plan_collection_"+suffix),
		WithEmbeddingTableName("plan_embedding_"+suffix),
		WithMetadataIndexes(index),
	)
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	t.Cleanup(func() {
		conn, err := pgx.Connect(context.Background(), url)
		if err != nil {
			return
		}
		defer conn.Close(context.Background())
		_, _ = conn.Exec(context.Background(),
			"DROP TABLE IF EXISTS "+store.embeddingTableName+", "+store.collectionTableName)
	})

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

	for _, flow := range []any{"3", 3, 1e6} {
		plan := explainSimilaritySearch(t, ctx, conn, store, "flow 3 document 1",
			map[string]any{"doc_type": "memory", "flow_id": flow})

		require.Contains(t, plan, index.indexName(store.embeddingTableName),
			"the metadata index must carry the scan for flow_id %v:\n%s", flow, plan)
		require.NotContains(t, plan, "Seq Scan on "+store.embeddingTableName,
			"the whole table was read for flow_id %v:\n%s", flow, plan)
	}
}

type recordingConn struct {
	PGXConn

	sql  string
	args []any
}

func (c *recordingConn) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	c.sql, c.args = sql, args
	return c.PGXConn.Query(ctx, sql, args...)
}

func explainSimilaritySearch(
	t *testing.T, ctx context.Context, conn *pgx.Conn, store Store, query string, filter map[string]any,
) string {
	t.Helper()

	recorder := &recordingConn{PGXConn: store.conn}
	store.conn = recorder
	_, err := store.SimilaritySearch(ctx, query, 3,
		vectorstores.WithScoreThreshold(0.2), vectorstores.WithFilters(filter))
	require.NoError(t, err)

	rows, err := conn.Query(ctx, "EXPLAIN (COSTS OFF) "+recorder.sql, recorder.args...)
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

func TestSimilaritySearchReadsATableWrittenByPythonLangchainPostgres(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()
	suffix := strings.ReplaceAll(uuid.New().String(), "-", "")
	collections, embeddings := "py_collection_"+suffix, "py_embedding_"+suffix

	conn, err := pgx.Connect(ctx, url)
	require.NoError(t, err)
	t.Cleanup(func() {
		_, _ = conn.Exec(context.Background(), "DROP TABLE IF EXISTS "+embeddings+", "+collections)
		_ = conn.Close(context.Background())
	})
	_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	require.NoError(t, err)
	_, err = conn.Exec(ctx, fmt.Sprintf(`CREATE TABLE %[1]s (uuid uuid PRIMARY KEY, name varchar NOT NULL UNIQUE, cmetadata json);
CREATE TABLE %[2]s (id varchar PRIMARY KEY, collection_id uuid REFERENCES %[1]s (uuid) ON DELETE CASCADE,
	embedding vector, document varchar, cmetadata jsonb)`, collections, embeddings))
	require.NoError(t, err)

	store, err := New(ctx, WithConnectionURL(url), WithEmbedder(fixedEmbedder{dims: 64}),
		WithCollectionName("py"), WithCollectionTableName(collections), WithEmbeddingTableName(embeddings))
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })

	var collectionID string
	require.NoError(t, conn.QueryRow(ctx, "SELECT uuid::text FROM "+collections+" WHERE name = 'py'").Scan(&collectionID))
	vector := fixedEmbedder{dims: 64}.vector("written by python")
	literal := make([]string, len(vector))
	for i, v := range vector {
		literal[i] = strconv.FormatFloat(float64(v), 'f', -1, 32)
	}
	for id, document := range []string{"written by python", "also written by python"} {
		_, err = conn.Exec(ctx, "INSERT INTO "+embeddings+" (id, collection_id, embedding, document, cmetadata) "+
			"VALUES ($1, $2, $3::vector, $4, '{}')", strconv.Itoa(id), collectionID, "["+strings.Join(literal, ",")+"]", document)
		require.NoError(t, err)
	}

	docs, err := store.SimilaritySearch(ctx, "written by python", 1)
	require.NoError(t, err, "the Python schema keys rows by id, not uuid")
	require.Len(t, docs, 1)
	require.Equal(t, "also written by python", docs[0].PageContent, "equal distances are ordered by document")
}

func newIsolatedIndexedStore(t *testing.T, url string, indexes ...MetadataIndex) (Store, *pgx.Conn) {
	t.Helper()

	ctx := t.Context()
	suffix := strings.ReplaceAll(uuid.New().String(), "-", "")
	embeddings, collections := "idx_embedding_"+suffix, "idx_collection_"+suffix
	conn, err := pgx.Connect(ctx, url)
	require.NoError(t, err)
	t.Cleanup(func() {
		_, _ = conn.Exec(context.Background(), "DROP TABLE IF EXISTS "+embeddings+", "+collections)
		_ = conn.Close(context.Background())
	})

	store, err := New(ctx, WithConnectionURL(url), WithEmbedder(fixedEmbedder{dims: 64}),
		WithCollectionName("c"), WithEmbeddingTableName(embeddings), WithCollectionTableName(collections),
		WithMetadataIndexes(indexes...))
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	return store, conn
}

func TestAnExcludeIndexServesTheStoresOwnFilter(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()
	index := MetadataIndex{Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "memory"}}
	store, conn := newIsolatedIndexedStore(t, url, index)

	predicates, args, err := filterPredicates("", map[string]any{"doc_type": "answer"}, 0)
	require.NoError(t, err)

	tx, err := conn.Begin(ctx)
	require.NoError(t, err)
	defer func() { _ = tx.Rollback(context.Background()) }()
	_, err = tx.Exec(ctx, "SET LOCAL enable_seqscan = off")
	require.NoError(t, err)
	rows, err := tx.Query(ctx, "EXPLAIN SELECT 1 FROM "+store.embeddingTableName+" WHERE "+predicates[0], args...)
	require.NoError(t, err)
	var plan []string
	for rows.Next() {
		var line string
		require.NoError(t, rows.Scan(&line))
		plan = append(plan, line)
	}
	require.NoError(t, rows.Err())

	require.Contains(t, strings.Join(plan, "\n"), index.indexName(store.embeddingTableName),
		"an equality filter on another value must be able to use the exclude index")
}

func TestAStartOverExistingIndexesTakesNoTableLock(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()
	store, conn := newIsolatedIndexedStore(t, url,
		MetadataIndex{Keys: []string{"doc_type"}},
		MetadataIndex{Keys: []string{"flow_id"}, Exclude: map[string]string{"flow_id": "0"}},
		MetadataIndex{Keys: []string{"owner"}, Name: "OwnerIdx_" + strings.ReplaceAll(uuid.New().String(), "-", "")})

	tx, err := conn.Begin(ctx)
	require.NoError(t, err)
	defer func() { _ = tx.Rollback(context.Background()) }()
	require.NoError(t, store.createMetadataIndexesIfNotExist(ctx, tx))

	rows, err := tx.Query(ctx, "SELECT mode FROM pg_locks WHERE locktype = 'relation' "+
		"AND relation = $1::regclass AND pid = pg_backend_pid()", store.embeddingTableName)
	require.NoError(t, err)
	var modes []string
	for rows.Next() {
		var mode string
		require.NoError(t, rows.Scan(&mode))
		modes = append(modes, mode)
	}
	require.NoError(t, rows.Err())
	require.Empty(t, modes, "indexes that exist need neither a build lock nor fresh statistics, so writers wait for nothing")
}

func TestSimilaritySearchChecksTheFilterBeforeTheDimensionGuard(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()
	store, conn := newIsolatedIndexedStore(t, url)
	_, err := store.AddDocuments(ctx, []schema.Document{{PageContent: "mine", Metadata: map[string]any{"flow_id": "3"}}})
	require.NoError(t, err)

	for _, flow := range []any{"3", 1e6} {
		plan := explainSimilaritySearch(t, ctx, conn, store, "mine", map[string]any{"flow_id": flow})

		require.Regexp(t, `cmetadata[^\n]*vector_dims`, plan,
			"a vector is detoasted for the dimension guard, so the filter on flow_id %v has to reject rows first:\n%s",
			flow, plan)
	}
}
