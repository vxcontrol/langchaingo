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

const verURL = "postgres://postgres@127.0.0.1:55471/postgres?sslmode=disable"

func TestProbeVerPartial(t *testing.T) {
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, verURL)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close(ctx)
	conn.Exec(ctx, "DROP TABLE IF EXISTS ver_part_emb, ver_part_coll")
	idx := MetadataIndex{Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "memory"}}
	store, err := New(ctx, WithConnectionURL(verURL), WithEmbedder(fixedEmbedder{dims: 8}),
		WithCollectionName("vp"), WithCollectionTableName("ver_part_coll"),
		WithEmbeddingTableName("ver_part_emb"), WithMetadataIndexes(idx))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	docs := make([]schema.Document, 0, 20000)
	for i := 0; i < 20000; i++ {
		dt := "memory"
		if i%100 == 0 {
			dt = "answer"
		}
		docs = append(docs, schema.Document{PageContent: fmt.Sprintf("d%d", i), Metadata: map[string]any{"doc_type": dt}})
	}
	if _, err := store.AddDocuments(ctx, docs); err != nil {
		t.Fatal(err)
	}
	conn.Exec(ctx, "ANALYZE ver_part_emb")
	var ddl string
	conn.QueryRow(ctx, "select indexdef from pg_indexes where tablename='ver_part_emb' and indexname like '%partial%'").Scan(&ddl)
	t.Logf("declared index: %s", ddl)
	explain := func(label string, seqOff bool) {
		rec := &recordingConn{PGXConn: store.conn}
		s2 := store
		s2.conn = rec
		got, err := s2.SimilaritySearch(ctx, "q", 3, vectorstores.WithFilters(map[string]any{"doc_type": "answer"}))
		if err != nil {
			t.Fatal(err)
		}
		tx, _ := conn.Begin(ctx)
		defer tx.Rollback(ctx)
		if seqOff {
			tx.Exec(ctx, "SET LOCAL enable_seqscan = off")
		}
		rows, err := tx.Query(ctx, "EXPLAIN (COSTS OFF) "+rec.sql, rec.args...)
		if err != nil {
			t.Fatal(err)
		}
		var b strings.Builder
		for rows.Next() {
			var l string
			rows.Scan(&l)
			if strings.Contains(l, "Scan") || strings.Contains(l, "Cond") || strings.Contains(l, "doc_type") {
				b.WriteString(l + "\n")
			}
		}
		rows.Close()
		t.Logf("%s (seqscan off=%v): results=%d\n%s", label, seqOff, len(got), b.String())
	}
	explain("IS DISTINCT FROM index only", false)
	explain("IS DISTINCT FROM index only", true)
	conn.Exec(ctx, "CREATE INDEX ver_part_ne ON ver_part_emb ((cmetadata ->> 'doc_type')) WHERE (cmetadata ->> 'doc_type') <> 'memory'")
	conn.Exec(ctx, "ANALYZE ver_part_emb")
	explain("plus <> index", false)
}
