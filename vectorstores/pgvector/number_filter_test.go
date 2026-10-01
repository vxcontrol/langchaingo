package pgvector

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
)

func TestANumberFilterDecodedFromJSONFindsItsDocument(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	for _, column := range []string{"json", "jsonb"} {
		t.Run(column, func(t *testing.T) {
			t.Parallel()

			ctx := t.Context()
			suffix := strings.ReplaceAll(uuid.New().String(), "-", "")
			collections, embeddings := "num_collection_"+suffix, "num_embedding_"+suffix
			conn, err := pgx.Connect(ctx, url)
			require.NoError(t, err)
			t.Cleanup(func() {
				_, _ = conn.Exec(context.Background(), "DROP TABLE IF EXISTS "+embeddings+", "+collections)
				_ = conn.Close(context.Background())
			})
			_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
			require.NoError(t, err)
			_, err = conn.Exec(ctx, fmt.Sprintf(`CREATE TABLE %[1]s (name varchar UNIQUE, cmetadata json,
	"uuid" uuid PRIMARY KEY);
CREATE TABLE %[2]s (collection_id uuid REFERENCES %[1]s (uuid) ON DELETE CASCADE, embedding vector,
	document varchar, cmetadata %[3]s, "uuid" uuid PRIMARY KEY)`, collections, embeddings, column))
			require.NoError(t, err)

			store, err := New(ctx, WithConnectionURL(url), WithEmbedder(fixedEmbedder{dims: 64}),
				WithCollectionName("c"), WithCollectionTableName(collections), WithEmbeddingTableName(embeddings))
			require.NoError(t, err)
			t.Cleanup(func() { _ = store.Close() })

			_, err = store.AddDocuments(ctx, []schema.Document{
				{PageContent: "a million", Metadata: map[string]any{"n": 1000000}},
				{PageContent: "two", Metadata: map[string]any{"n": 2}},
				{PageContent: "five written as text", Metadata: map[string]any{"n": "5"}},
			})
			require.NoError(t, err)

			for filter, want := range map[string]string{
				`{"n": 1e6}`:     "a million",
				`{"n": 1000000}`: "a million",
				`{"n": 5}`:       "five written as text",
			} {
				var decoded map[string]any
				require.NoError(t, json.Unmarshal([]byte(filter), &decoded))

				docs, err := store.SimilaritySearch(ctx, "anything", 10, vectorstores.WithFilters(decoded))
				require.NoError(t, err)
				require.Len(t, docs, 1, filter)
				require.Equal(t, want, docs[0].PageContent, filter)
			}
		})
	}
}
