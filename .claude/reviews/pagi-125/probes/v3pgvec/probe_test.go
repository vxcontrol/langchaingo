package pgvector

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

type recTx struct {
	pgx.Tx
	stmts []string
}

func (r *recTx) Exec(_ context.Context, sql string, _ ...any) (pgconn.CommandTag, error) {
	r.stmts = append(r.stmts, sql)
	return pgconn.CommandTag{}, nil
}

func TestProbeFilterKeys(t *testing.T) {
	for _, k := range []string{"doc_id", "doc-id", "file.name", "x-tenant-id", "ключ"} {
		p, a, err := filterPredicates("data.", map[string]any{k: "42"}, 3)
		t.Logf("key=%q -> predicates=%v args=%v err=%v", k, p, a, err)
	}
}

func TestProbeAnalyzeEveryStart(t *testing.T) {
	s := Store{embeddingTableName: "langchain_pg_embedding", metadataIndexes: []MetadataIndex{{Keys: []string{"tenant"}}}}
	for start := 1; start <= 2; start++ {
		tx := &recTx{}
		if err := s.createMetadataIndexesIfNotExist(context.Background(), tx); err != nil {
			t.Fatal(err)
		}
		t.Logf("start %d statements: %q", start, tx.stmts)
	}
}
