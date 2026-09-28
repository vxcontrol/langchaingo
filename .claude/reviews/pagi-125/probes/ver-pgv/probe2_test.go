package pgvector

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/vxcontrol/langchaingo/schema"
)

func TestProbeVerAnalyze(t *testing.T) {
	url := "postgres://postgres@127.0.0.1:55471/postgres?sslmode=disable"
	ctx := context.Background()
	conn, _ := pgx.Connect(ctx, url)
	defer conn.Close(ctx)
	conn.Exec(ctx, "DROP TABLE IF EXISTS ver_an_emb, ver_an_coll")
	var ver string
	conn.QueryRow(ctx, "select extversion from pg_extension where extname='vector'").Scan(&ver)
	t.Logf("pgvector %s", ver)
	common := []Option{WithConnectionURL(url), WithCollectionName("c"),
		WithCollectionTableName("ver_an_coll"), WithEmbeddingTableName("ver_an_emb")}
	for _, d := range []int{64, 32} {
		s, err := New(ctx, append(common, WithEmbedder(fixedEmbedder{dims: d}))...)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = s.AddDocuments(ctx, []schema.Document{{PageContent: "a", Metadata: map[string]any{"flow_id": "1"}}, {PageContent: "b", Metadata: map[string]any{"flow_id": "2"}}}); err != nil {
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
	var n int
	conn.QueryRow(ctx, "select count(*) from pg_indexes where tablename='ver_an_emb' and indexname like '%meta%'").Scan(&n)
	t.Logf("metadata indexes present after failed New: %d", n)
}
