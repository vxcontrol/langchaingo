package pgvector

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/schema"
)

func TestEqualMatchesAreOrderedByTheirMetadataWhateverTheTableLayout(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	for _, column := range []string{"json", "jsonb"} {
		t.Run(column, func(t *testing.T) {
			t.Parallel()

			ctx := t.Context()
			store, conn := newIsolatedIndexedStore(t, url)
			_, err := conn.Exec(ctx, "ALTER TABLE "+store.embeddingTableName+
				" ALTER COLUMN cmetadata TYPE "+column+" USING cmetadata::"+column)
			require.NoError(t, err)

			_, err = store.AddDocuments(ctx, []schema.Document{
				{PageContent: "same text", Metadata: map[string]any{"source": "b"}},
				{PageContent: "same text", Metadata: map[string]any{"source": "c"}},
				{PageContent: "same text", Metadata: map[string]any{"source": "a"}},
			})
			require.NoError(t, err)

			for _, moved := range []string{"", "a", "b", "c", "a"} {
				if moved != "" {
					_, err = conn.Exec(ctx, "UPDATE "+store.embeddingTableName+
						" SET document = document WHERE cmetadata->>'source' = $1", moved)
					require.NoError(t, err)
					_, err = conn.Exec(ctx, "VACUUM FULL "+store.embeddingTableName)
					require.NoError(t, err)
				}

				docs, err := store.SimilaritySearch(ctx, "same text", 3)
				require.NoError(t, err)
				sources := make([]any, 0, len(docs))
				for _, doc := range docs {
					sources = append(sources, doc.Metadata["source"])
				}
				require.Equal(t, []any{"a", "b", "c"}, sources, "after moving %q to the end of the table", moved)
			}
		})
	}
}
