package pgvector

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
)

type score float64

func TestANumberFilterFindsTheDocumentsHoldingItsValue(t *testing.T) {
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
				{PageContent: "a million", Metadata: map[string]any{"n": 1000000}},
				{PageContent: "a million in Go text", Metadata: map[string]any{"n": fmt.Sprint(1e6)}},
				{PageContent: "two", Metadata: map[string]any{"n": 2}},
				{PageContent: "five in text", Metadata: map[string]any{"n": "5"}},
				{PageContent: "huge", Metadata: map[string]any{"n": 1e21}},
				{PageContent: "tiny", Metadata: map[string]any{"n": 1e-7}},
				{PageContent: "an id past float precision", Metadata: map[string]any{"n": uint64(12345678901234567000)}},
			})
			require.NoError(t, err)

			for name, tc := range map[string]struct {
				filter any
				want   []string
			}{
				"float decoded from JSON":         {1e6, []string{"a million", "a million in Go text"}},
				"int":                             {1000000, []string{"a million"}},
				"named float type":                {score(1e6), []string{"a million", "a million in Go text"}},
				"json.Number":                     {json.Number("1e6"), []string{"a million"}},
				"number held as text":             {5, []string{"five in text"}},
				"float from 1e21 up":              {1e21, []string{"huge"}},
				"float below 1e-6":                {1e-7, []string{"tiny"}},
				"json.Number that is no number":   {json.Number(""), nil},
				"uint64":                          {uint64(12345678901234567000), []string{"an id past float precision"}},
				"json.Number of another id":       {json.Number("12345678901234567890"), nil},
				"named int with a String method":  {time.Duration(2), []string{"two"}},
				"named uint with a String method": {os.FileMode(2), []string{"two"}},
			} {
				docs, err := store.SimilaritySearch(ctx, "anything", 10,
					vectorstores.WithFilters(map[string]any{"n": tc.filter}))
				require.NoError(t, err, name)
				found := make([]string, 0, len(docs))
				for _, doc := range docs {
					found = append(found, doc.PageContent)
				}
				require.ElementsMatch(t, tc.want, found, name)
			}
		})
	}
}
