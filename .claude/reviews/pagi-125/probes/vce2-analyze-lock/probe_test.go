package pgvector_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/vxcontrol/langchaingo/vectorstores/pgvector"
)

const vce2lURL = "postgres://postgres@127.0.0.1:55491/postgres?sslmode=disable"

type vce2lEmb struct{}

func (vce2lEmb) EmbedDocuments(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		v := make([]float32, 1536)
		v[0] = 1
		out[i] = v
	}
	return out, nil
}
func (vce2lEmb) EmbedQuery(context.Context, string) ([]float32, error) {
	v := make([]float32, 1536)
	v[0] = 1
	return v, nil
}

func vce2lOpen(ctx context.Context, withIdx bool) (time.Duration, error) {
	opts := []pgvector.Option{pgvector.WithConnectionURL(vce2lURL), pgvector.WithEmbedder(vce2lEmb{}),
		pgvector.WithCollectionName("big"), pgvector.WithCollectionTableName("vce2l_coll"),
		pgvector.WithEmbeddingTableName("vce2l_emb")}
	if withIdx {
		opts = append(opts, pgvector.WithMetadataIndexes(pgvector.MetadataIndex{Keys: []string{"flow_id"}}))
	}
	start := time.Now()
	s, err := pgvector.New(ctx, opts...)
	d := time.Since(start)
	if err == nil {
		_ = s.Close()
	}
	return d, err
}

func TestProbeVce2AnalyzeLock(t *testing.T) {
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, vce2lURL)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close(ctx)
	var n int
	_ = conn.QueryRow(ctx, "SELECT count(*) FROM vce2l_emb").Scan(&n)
	if n == 0 {
		if _, err := vce2lOpen(ctx, true); err != nil {
			t.Fatal(err)
		}
		_, err = conn.Exec(ctx, `INSERT INTO vce2l_emb (collection_id, embedding, document, cmetadata, uuid)
			SELECT (SELECT uuid FROM vce2l_coll WHERE name='big'), array_fill(random()::real, ARRAY[1536])::vector,
			'doc '||i, json_build_object('flow_id', (i % 100)::text), gen_random_uuid()
			FROM generate_series(1, 40000) i`)
		if err != nil {
			t.Fatal(err)
		}
		_ = conn.QueryRow(ctx, "SELECT count(*) FROM vce2l_emb").Scan(&n)
	}
	t.Logf("rows: %d", n)

	for _, withIdx := range []bool{false, true, false, true} {
		var wg sync.WaitGroup
		var insertTook time.Duration
		var insertErr error
		wg.Add(1)
		go func() {
			defer wg.Done()
			w, err := pgx.Connect(ctx, vce2lURL)
			if err != nil {
				insertErr = err
				return
			}
			defer w.Close(ctx)
			time.Sleep(30 * time.Millisecond)
			start := time.Now()
			_, insertErr = w.Exec(ctx, `INSERT INTO vce2l_emb (collection_id, embedding, document, cmetadata, uuid)
				SELECT uuid, array_fill(0.5::real, ARRAY[1536])::vector, 'w', '{"flow_id":"1"}', gen_random_uuid() FROM vce2l_coll WHERE name='big'`)
			insertTook = time.Since(start)
		}()
		openTook, err := vce2lOpen(ctx, withIdx)
		wg.Wait()
		t.Logf("WithMetadataIndexes=%v: New took %v (err=%v); concurrent INSERT took %v (err=%v)",
			withIdx, openTook, err, insertTook, insertErr)
	}
}
