package pgvector

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
)

func TestProbeFloatFilter(t *testing.T) {
	url := "postgres://postgres@127.0.0.1:55439/postgres?sslmode=disable"
	ctx := context.Background()
	conn, _ := pgx.Connect(ctx, url)
	defer conn.Close(ctx)
	defer conn.Exec(ctx, "DROP TABLE IF EXISTS probe_ff_emb, probe_ff_coll")
	s, err := New(ctx, WithConnectionURL(url), WithCollectionName("c"), WithEmbedder(fixedEmbedder{dims: 8}),
		WithCollectionTableName("probe_ff_coll"), WithEmbeddingTableName("probe_ff_emb"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	_, err = s.AddDocuments(ctx, []schema.Document{{PageContent: "mine", Metadata: map[string]any{"user_id": 1234567}}})
	if err != nil {
		t.Fatal(err)
	}
	// A filter decoded from a JSON request body, as an HTTP API would receive it.
	var filter map[string]any
	_ = json.Unmarshal([]byte(`{"user_id": 1234567}`), &filter)
	docs, err := s.SimilaritySearch(ctx, "q", 5, vectorstores.WithFilters(filter))
	t.Logf("json-decoded filter %#v -> %d docs err=%v", filter["user_id"], len(docs), err)
	docs, err = s.SimilaritySearch(ctx, "q", 5, vectorstores.WithFilters(map[string]any{"user_id": 1234567}))
	t.Logf("int filter -> %d docs err=%v", len(docs), err)
	_, args, _ := filterPredicates("", filter, 0)
	t.Logf("bound value for float64: %q", args[0])
}
