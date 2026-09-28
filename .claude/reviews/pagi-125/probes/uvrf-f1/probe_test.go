package pgvector_test

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/vxcontrol/langchaingo/vectorstores"
	"github.com/vxcontrol/langchaingo/vectorstores/pgvector"
)

const uvf1URL = "postgres://postgres@127.0.0.1:55497/postgres?sslmode=disable"

type uvf1Emb struct{}

func (uvf1Emb) EmbedDocuments(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1, 0, 0}
	}
	return out, nil
}
func (uvf1Emb) EmbedQuery(context.Context, string) ([]float32, error) { return []float32{1, 0, 0}, nil }

// Tables laid out the way langchain_postgres (Python) creates them: the
// embedding table's key column is "id", not "uuid", and cmetadata is jsonb.
func TestProbeUvf1PythonSchema(t *testing.T) {
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, uvf1URL)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close(ctx)
	for _, q := range []string{
		"DROP TABLE IF EXISTS uvf1_py_emb, uvf1_py_coll",
		"CREATE EXTENSION IF NOT EXISTS vector",
		"CREATE TABLE uvf1_py_coll (uuid uuid PRIMARY KEY, name varchar NOT NULL UNIQUE, cmetadata json)",
		"CREATE TABLE uvf1_py_emb (id varchar PRIMARY KEY, collection_id uuid REFERENCES uvf1_py_coll(uuid) ON DELETE CASCADE, embedding vector, document varchar, cmetadata jsonb)",
		"INSERT INTO uvf1_py_coll VALUES ('00000000-0000-0000-0000-000000000001', 'pycol', '{}')",
		"INSERT INTO uvf1_py_emb VALUES ('a', '00000000-0000-0000-0000-000000000001', '[1,0,0]', 'doc a', '{\"flow_id\":\"1\"}')",
	} {
		if _, err := conn.Exec(ctx, q); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	store, err := pgvector.New(ctx, pgvector.WithConnectionURL(uvf1URL), pgvector.WithEmbedder(uvf1Emb{}),
		pgvector.WithCollectionName("pycol"), pgvector.WithCollectionTableName("uvf1_py_coll"),
		pgvector.WithEmbeddingTableName("uvf1_py_emb"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	docs, err := store.SimilaritySearch(ctx, "q", 3)
	t.Logf("SimilaritySearch: docs=%d err=%v", len(docs), err)
	docs, err = store.SimilaritySearch(ctx, "q", 3, vectorstores.WithFilters(map[string]any{"flow_id": "1"}))
	t.Logf("SimilaritySearch filtered: docs=%d err=%v", len(docs), err)
}
