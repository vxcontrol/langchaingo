package pgvector

import (
	"context"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/schema"
)

func TestAStoreOnSchemaQualifiedTablesStarts(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	ctx := t.Context()
	schemaName := "lcg_" + strings.ReplaceAll(uuid.New().String(), "-", "")
	conn, err := pgx.Connect(ctx, url)
	require.NoError(t, err)
	t.Cleanup(func() {
		_, _ = conn.Exec(context.Background(), "DROP SCHEMA IF EXISTS "+schemaName+" CASCADE")
		_ = conn.Close(context.Background())
	})
	_, err = conn.Exec(ctx, "CREATE SCHEMA "+schemaName)
	require.NoError(t, err)

	store, err := New(ctx, WithConnectionURL(url), WithEmbedder(fixedEmbedder{dims: 64}),
		WithCollectionName("c"), WithCollectionTableName(schemaName+".collection"),
		WithEmbeddingTableName(schemaName+".embedding"), WithVectorDimensions(64),
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

func TestAMetadataIndexNamedLikeAReservedWordIsCreated(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	store, conn := newIsolatedIndexedStore(t, url, MetadataIndex{Name: "Order", Keys: []string{"flow_id"}})

	var exists bool
	require.NoError(t, conn.QueryRow(t.Context(),
		"SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = $1 AND indexname = 'order')",
		store.embeddingTableName).Scan(&exists))
	require.True(t, exists)
}
