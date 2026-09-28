package pgvector

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
)

func TestProbePartialIndexReachable(t *testing.T) {
	url := "postgres://postgres@127.0.0.1:55439/postgres?sslmode=disable"
	ctx := context.Background()
	idx := MetadataIndex{Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "memory"}}
	store, err := New(ctx,
		WithConnectionURL(url),
		WithEmbedder(fixedEmbedder{dims: 8}),
		WithCollectionName("probe-partial"),
		WithCollectionTableName("probe_partial_coll"),
		WithEmbeddingTableName("probe_partial_emb"),
		WithMetadataIndexes(idx),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	conn, _ := pgx.Connect(ctx, url)
	defer conn.Close(ctx)
	defer conn.Exec(ctx, "DROP TABLE IF EXISTS probe_partial_emb, probe_partial_coll")

	docs := make([]schema.Document, 0, 5000)
	for i := 0; i < 5000; i++ {
		dt := "memory"
		if i%100 == 0 {
			dt = "answer"
		}
		docs = append(docs, schema.Document{PageContent: fmt.Sprintf("d%d", i), Metadata: map[string]any{"doc_type": dt}})
	}
	if _, err := store.AddDocuments(ctx, docs); err != nil {
		t.Fatal(err)
	}
	conn.Exec(ctx, "ANALYZE probe_partial_emb")

	explain := func(label string) {
		rec := &recordingConn{PGXConn: store.conn}
		s2 := store
		s2.conn = rec
		got, err := s2.SimilaritySearch(ctx, "q", 3, vectorstores.WithFilters(map[string]any{"doc_type": "answer"}))
		if err != nil {
			t.Fatal(err)
		}
		tx, _ := conn.Begin(ctx)
		defer tx.Rollback(ctx)
		tx.Exec(ctx, "SET LOCAL enable_seqscan = off")
		rows, err := tx.Query(ctx, "EXPLAIN (COSTS OFF) "+rec.sql, rec.args...)
		if err != nil {
			t.Fatal(err)
		}
		var b strings.Builder
		for rows.Next() {
			var l string
			rows.Scan(&l)
			b.WriteString(l + "\n")
		}
		rows.Close()
		t.Logf("%s: results=%d\n%s", label, len(got), b.String())
	}
	t.Logf("DDL: %s", func() string { s, _ := idx.ddl("probe_partial_emb"); return s }())
	explain("declared partial index (IS DISTINCT FROM)")
	conn.Exec(ctx, "CREATE INDEX probe_partial_ne ON probe_partial_emb ((cmetadata ->> 'doc_type')) WHERE (cmetadata ->> 'doc_type') <> 'memory'")
	conn.Exec(ctx, "ANALYZE probe_partial_emb")
	explain("same index with <> predicate")
}
