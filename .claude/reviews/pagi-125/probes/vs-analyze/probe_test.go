package pgvector

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/vxcontrol/langchaingo/schema"
)

func TestProbeAnalyzeMixedDims(t *testing.T) {
	url := "postgres://postgres@127.0.0.1:55439/postgres?sslmode=disable"
	ctx := context.Background()
	conn, _ := pgx.Connect(ctx, url)
	defer conn.Close(ctx)
	defer conn.Exec(ctx, "DROP TABLE IF EXISTS probe_an_emb, probe_an_coll")
	var ver string
	conn.QueryRow(ctx, "select extversion from pg_extension where extname='vector'").Scan(&ver)
	t.Logf("pgvector %s", ver)
	common := []Option{WithConnectionURL(url), WithCollectionName("c"),
		WithCollectionTableName("probe_an_coll"), WithEmbeddingTableName("probe_an_emb")}
	for _, d := range []int{64, 32} {
		s, err := New(ctx, append(common, WithEmbedder(fixedEmbedder{dims: d}))...)
		if err != nil {
			t.Fatal(err)
		}
		_, err = s.AddDocuments(ctx, []schema.Document{{PageContent: "a", Metadata: map[string]any{"flow_id": "1"}}, {PageContent: "b", Metadata: map[string]any{"flow_id": "2"}}})
		if err != nil {
			t.Fatal(err)
		}
		s.Close()
	}
	s, err := New(ctx, append(common, WithEmbedder(fixedEmbedder{dims: 32}))...)
	t.Logf("New without metadata indexes: err=%v", err)
	if err == nil {
		s.Close()
	}
	s, err = New(ctx, append(common, WithEmbedder(fixedEmbedder{dims: 32}),
		WithMetadataIndexes(MetadataIndex{Keys: []string{"flow_id"}}))...)
	t.Logf("New with WithMetadataIndexes: err=%v", err)
	if err == nil {
		s.Close()
	}
}
