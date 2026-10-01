package pgvector

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/schema"
)

func newSchema(t *testing.T, url string) (string, *pgx.Conn) {
	t.Helper()

	ctx := t.Context()
	name := "lcg_" + strings.ReplaceAll(uuid.New().String(), "-", "")
	conn, err := pgx.Connect(ctx, url)
	require.NoError(t, err)
	t.Cleanup(func() {
		_, _ = conn.Exec(context.Background(), "DROP SCHEMA IF EXISTS "+name+" CASCADE")
		_ = conn.Close(context.Background())
	})
	_, err = conn.Exec(ctx, "CREATE SCHEMA "+name)
	require.NoError(t, err)
	return name, conn
}

func indexDefinitions(t *testing.T, conn *pgx.Conn, schemaName, table string) []string {
	t.Helper()

	rows, err := conn.Query(t.Context(),
		"SELECT indexdef FROM pg_indexes WHERE schemaname = $1 AND tablename = $2", schemaName, table)
	require.NoError(t, err)
	definitions, err := pgx.CollectRows(rows, pgx.RowTo[string])
	require.NoError(t, err)
	return definitions
}

func TestAStoreOnSchemaQualifiedTablesKeepsOneSetOfIndexes(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()
	schemaName, conn := newSchema(t, url)
	const table = "langchain_pg_embedding_tenant"

	open := func(url, prefix string) {
		store, err := New(ctx, WithConnectionURL(url), WithEmbedder(fixedEmbedder{dims: 64}),
			WithCollectionName("c"), WithCollectionTableName(prefix+"collection"),
			WithEmbeddingTableName(prefix+table), WithVectorDimensions(64),
			WithHNSWIndex(16, 64, "vector_cosine_ops"),
			WithMetadataIndexes(MetadataIndex{Keys: []string{"flow_id"}}))
		require.NoError(t, err)
		t.Cleanup(func() { _ = store.Close() })

		_, err = store.AddDocuments(ctx, []schema.Document{{PageContent: "mine", Metadata: map[string]any{"flow_id": "1"}}})
		require.NoError(t, err)
		docs, err := store.SimilaritySearch(ctx, "mine", 1)
		require.NoError(t, err)
		require.Len(t, docs, 1)
	}
	open(url, schemaName+".")
	var searchPath string
	require.NoError(t, conn.QueryRow(ctx, "SHOW search_path").Scan(&searchPath))
	open(withParameter(t, url, "search_path", schemaName+", "+searchPath), "")

	definitions := indexDefinitions(t, conn, schemaName, table)
	require.Len(t, definitions, 4, "the primary key, collection_id, HNSW and metadata indexes, once each:\n%s",
		strings.Join(definitions, "\n"))
	require.Contains(t, strings.Join(definitions, "\n"), "USING hnsw")
}

func TestEverySpellingOfATableSharesOneSetOfIndexes(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()
	schemaName, conn := newSchema(t, url)
	upperSchema := strings.ToUpper(schemaName)
	for relation, spellings := range map[string][]string{
		"emb": {`emb`, `"emb"`, `EMB`, ` emb `, upperSchema + `."emb"`},
		"Emb": {`"Emb"`},
		"a.b": {`"a.b"`},
		"b":   {`b`},
		`q"t`: {`"q""t"`},
		"qt":  {`qt`},
	} {
		for _, spelling := range spellings {
			table := spelling
			if !strings.Contains(spelling, ".") || strings.HasPrefix(spelling, `"`) {
				table = schemaName + "." + spelling
			}
			store, err := New(ctx, WithConnectionURL(url), WithEmbedder(fixedEmbedder{dims: 64}),
				WithCollectionName("c"), WithCollectionTableName(schemaName+".collection"),
				WithEmbeddingTableName(table), WithVectorDimensions(64),
				WithHNSWIndex(16, 64, "vector_cosine_ops"),
				WithMetadataIndexes(MetadataIndex{Keys: []string{"flow_id"}}))
			require.NoError(t, err, table)
			t.Cleanup(func() { _ = store.Close() })
			require.Equal(t, relation, store.embeddingRelation, table)

			_, err = store.AddDocuments(ctx, []schema.Document{{PageContent: table, Metadata: map[string]any{"flow_id": "1"}}})
			require.NoError(t, err, table)
			docs, err := store.SimilaritySearch(ctx, table, 1)
			require.NoError(t, err, table)
			require.Len(t, docs, 1, table)
		}
		definitions := indexDefinitions(t, conn, schemaName, relation)
		require.Len(t, definitions, 4, "the primary key, collection_id, HNSW and metadata indexes of %s:\n%s",
			relation, strings.Join(definitions, "\n"))
	}
}

func TestAMetadataIndexNamedLikeAReservedWordIsCreated(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	schemaName, conn := newSchema(t, url)
	store, err := New(t.Context(), WithConnectionURL(url), WithEmbedder(fixedEmbedder{dims: 64}),
		WithCollectionName("c"), WithCollectionTableName(schemaName+".collection"),
		WithEmbeddingTableName(schemaName+".embedding"),
		WithMetadataIndexes(MetadataIndex{Name: "Order", Keys: []string{"flow_id"}}))
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })

	var exists bool
	require.NoError(t, conn.QueryRow(t.Context(),
		"SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE schemaname = $1 AND indexname = 'order')",
		schemaName).Scan(&exists))
	require.True(t, exists)
}

func TestAStoreAdoptsTheMetadataIndexItsUnquotedNameCreated(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()
	schemaName, conn := newSchema(t, url)
	const table = "Émbed"
	index := MetadataIndex{Keys: []string{"flow_id"}}

	open := func(url, table string, indexes ...MetadataIndex) {
		store, err := New(ctx, WithConnectionURL(url), WithEmbedder(fixedEmbedder{dims: 64}),
			WithCollectionName("c"), WithCollectionTableName(schemaName+".collection"),
			WithEmbeddingTableName(table), WithMetadataIndexes(indexes...))
		require.NoError(t, err)
		require.NoError(t, store.Close())
	}
	open(url, schemaName+"."+table)
	_, err := conn.Exec(ctx, fmt.Sprintf("CREATE INDEX %s_meta_flow_id_%08x ON %s.%s ((cmetadata ->> 'flow_id'))",
		table, index.fingerprint(table), schemaName, table))
	require.NoError(t, err)
	var searchPath string
	require.NoError(t, conn.QueryRow(ctx, "SHOW search_path").Scan(&searchPath))
	open(withParameter(t, url, "search_path", schemaName+", "+searchPath), table, index)

	var metadataIndexes int
	require.NoError(t, conn.QueryRow(ctx, `SELECT count(*) FROM pg_indexes
		WHERE schemaname = $1 AND indexdef LIKE '%cmetadata%'`, schemaName).Scan(&metadataIndexes))
	require.Equal(t, 1, metadataIndexes)
}
