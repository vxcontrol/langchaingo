package pgvector_test

import (
	"context"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/vxcontrol/langchaingo/vectorstores/pgvector"
)

const uvf3URL = "postgres://postgres@127.0.0.1:55497/postgres?sslmode=disable"

type uvf3Emb struct{}

func (uvf3Emb) EmbedDocuments(_ context.Context, texts []string) ([][]float32, error) {
	return make([][]float32, len(texts)), nil
}
func (uvf3Emb) EmbedQuery(context.Context, string) ([]float32, error) { return nil, nil }

func TestProbeUvf3WriterBlocked(t *testing.T) {
	ctx := context.Background()
	for _, withIdx := range []bool{false, true, false, true, false, true} {
		w, err := pgx.Connect(ctx, uvf3URL)
		if err != nil {
			t.Fatal(err)
		}
		opts := []pgvector.Option{pgvector.WithConnectionURL(uvf3URL), pgvector.WithEmbedder(uvf3Emb{}),
			pgvector.WithCollectionName("c"), pgvector.WithCollectionTableName("uvf2_coll"),
			pgvector.WithEmbeddingTableName("uvf2_emb")}
		if withIdx {
			opts = append(opts, withIdxOpt())
		}
		done := make(chan time.Duration, 1)
		go func() {
			start := time.Now()
			s, err := pgvector.New(ctx, opts...)
			if err != nil {
				t.Error(err)
			} else {
				_ = s.Close()
			}
			done <- time.Since(start)
		}()
		// Wait until some other backend holds ShareLock on the table, then insert.
		seen := false
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			var n int
			_ = w.QueryRow(ctx, `SELECT count(*) FROM pg_locks WHERE relation='uvf2_emb'::regclass AND mode='ShareLock' AND granted AND pid<>pg_backend_pid()`).Scan(&n)
			if n > 0 {
				seen = true
				break
			}
			select {
			case d := <-done:
				done <- d
				deadline = time.Now()
			default:
			}
		}
		start := time.Now()
		_, ierr := w.Exec(ctx, `INSERT INTO uvf2_emb (collection_id, embedding, document, cmetadata, uuid) VALUES ('00000000-0000-0000-0000-000000000001','[1,2]','w','{}',gen_random_uuid())`)
		ins := time.Since(start)
		newTook := <-done
		t.Logf("metadataIndexes=%v: ShareLock observed=%v; New took %v; INSERT issued while lock held took %v (err=%v)", withIdx, seen, newTook, ins, ierr)
		_, _ = w.Exec(ctx, `DELETE FROM uvf2_emb WHERE document='w'`)
		w.Close(ctx)
	}
}
