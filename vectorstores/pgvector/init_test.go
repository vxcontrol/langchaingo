package pgvector

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/stretchr/testify/require"
)

type scriptedTx struct {
	pgx.Tx

	failOn   string
	failWith error

	committed      bool
	rolledBack     bool
	rollbackCtxErr error
}

func (tx *scriptedTx) fails(sql string) bool {
	return tx.failOn != "" && strings.Contains(sql, tx.failOn)
}

func (tx *scriptedTx) Exec(_ context.Context, sql string, _ ...any) (pgconn.CommandTag, error) {
	if tx.fails(sql) {
		return pgconn.CommandTag{}, tx.failWith
	}
	return pgconn.CommandTag{}, nil
}

func (tx *scriptedTx) QueryRow(_ context.Context, sql string, _ ...any) pgx.Row {
	if tx.fails(sql) {
		return scriptedRow{err: tx.failWith}
	}
	return scriptedRow{}
}

func (tx *scriptedTx) Commit(context.Context) error {
	tx.committed = true
	return nil
}

func (tx *scriptedTx) Rollback(ctx context.Context) error {
	tx.rolledBack = true
	tx.rollbackCtxErr = ctx.Err()
	return nil
}

type scriptedRow struct{ err error }

func (r scriptedRow) Scan(dest ...any) error {
	if r.err != nil {
		return r.err
	}
	if id, ok := dest[0].(*string); ok {
		*id = uuid.NewString()
	}
	return nil
}

type scriptedConn struct {
	PGXConn

	tx *scriptedTx
}

func (scriptedConn) Ping(context.Context) error { return nil }

func (c scriptedConn) Begin(context.Context) (pgx.Tx, error) { return c.tx, nil }

func openScripted(ctx context.Context, tx *scriptedTx) error {
	_, err := New(ctx,
		WithConn(scriptedConn{tx: tx}),
		WithEmbedder(fixedEmbedder{dims: 3}),
		WithPreDeleteCollection(true),
		WithHNSWIndex(16, 64, "vector_l2_ops"),
		WithMetadataIndexes(MetadataIndex{Keys: []string{"flow_id"}}),
	)
	return err
}

func TestInitRollsBackWhenAnyStepFails(t *testing.T) {
	t.Parallel()

	failure := errors.New("step failed")
	for _, step := range []string{
		"pg_advisory_xact_lock",
		"CREATE EXTENSION",
		"CREATE TABLE IF NOT EXISTS langchain_pg_collection",
		"CREATE TABLE IF NOT EXISTS langchain_pg_embedding",
		"CREATE INDEX IF NOT EXISTS langchain_pg_embedding_collection_id",
		"CREATE INDEX IF NOT EXISTS langchain_pg_embedding_embedding_hnsw",
		"CREATE INDEX IF NOT EXISTS langchain_pg_embedding_meta_flow_id",
		"ANALYZE",
		"DELETE FROM langchain_pg_collection",
		"INSERT INTO langchain_pg_collection",
		"SELECT uuid FROM langchain_pg_collection",
	} {
		t.Run(step, func(t *testing.T) {
			t.Parallel()

			tx := &scriptedTx{failOn: step, failWith: failure}
			require.ErrorIs(t, openScripted(t.Context(), tx), failure)
			require.True(t, tx.rolledBack)
			require.False(t, tx.committed)
		})
	}
}

func TestInitCommitsWithoutRollingBack(t *testing.T) {
	t.Parallel()

	tx := &scriptedTx{}
	require.NoError(t, openScripted(t.Context(), tx))
	require.True(t, tx.committed)
	require.False(t, tx.rolledBack)
}

func TestInitRollsBackEvenWhenTheCallerCancelled(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	tx := &scriptedTx{failOn: "CREATE EXTENSION", failWith: context.Canceled}
	require.ErrorIs(t, openScripted(ctx, tx), context.Canceled)
	require.True(t, tx.rolledBack)
	require.NoError(t, tx.rollbackCtxErr)
}

type cancelAfterQuery struct {
	marker string
	cancel context.CancelFunc
	armed  bool
}

func (c *cancelAfterQuery) TraceQueryStart(
	ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData,
) context.Context {
	c.armed = strings.Contains(data.SQL, c.marker)
	return ctx
}

func (c *cancelAfterQuery) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {
	if c.armed {
		c.cancel()
	}
}

func TestInitCancelledBeforeCommitKeepsTheCallersConnection(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	collectionTable := "cancel_collection_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:12]

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	cfg, err := pgx.ParseConfig(url)
	require.NoError(t, err)
	cfg.Tracer = &cancelAfterQuery{marker: "SELECT uuid FROM " + collectionTable, cancel: cancel}
	conn, err := pgx.ConnectConfig(t.Context(), cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close(context.Background()) })

	_, err = New(ctx,
		WithConn(conn),
		WithEmbedder(fixedEmbedder{dims: 3}),
		WithCollectionName("cancel"),
		WithCollectionTableName(collectionTable),
		WithEmbeddingTableName("cancel_embedding_"+collectionTable[len("cancel_collection_"):]),
	)
	require.ErrorIs(t, err, context.Canceled)
	require.False(t, conn.IsClosed(), "a caller that cancelled before the commit lost its connection")

	var created bool
	require.NoError(t, conn.QueryRow(t.Context(), "SELECT to_regclass($1) IS NOT NULL", collectionTable).Scan(&created))
	require.False(t, created, "a cancelled init must roll back rather than commit")
}

// Not parallel: a store opening in parallel holds the advisory locks this timeout also bounds.
func TestFailedInitReturnsItsConnectionToThePool(t *testing.T) {
	url := narrowingURL(t)
	ctx := t.Context()

	suffix := strings.ReplaceAll(uuid.NewString(), "-", "")[:12]
	collectionTable := "rollback_collection_" + suffix
	embeddingTable := "rollback_embedding_" + suffix

	cfg, err := pgxpool.ParseConfig(url)
	require.NoError(t, err)
	cfg.MaxConns = 1
	cfg.ConnConfig.RuntimeParams["statement_timeout"] = "500"
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	require.NoError(t, err)
	t.Cleanup(func() {
		closed := make(chan struct{})
		go func() {
			pool.Close()
			close(closed)
		}()
		select {
		case <-closed:
		case <-time.After(5 * time.Second):
			t.Error("closing the pool waits on a connection that was never released")
		}
	})

	open := func(ctx context.Context) error {
		_, err := New(ctx,
			WithConn(pool),
			WithEmbedder(fixedEmbedder{dims: 3}),
			WithCollectionName("rollback"),
			WithCollectionTableName(collectionTable),
			WithEmbeddingTableName(embeddingTable),
		)
		return err
	}
	require.NoError(t, open(ctx))
	t.Cleanup(func() {
		conn, err := pgx.Connect(context.Background(), url)
		if err != nil {
			return
		}
		defer conn.Close(context.Background())
		_, _ = conn.Exec(context.Background(), "DROP TABLE IF EXISTS "+embeddingTable+", "+collectionTable)
	})

	writer, err := pgx.Connect(ctx, url)
	require.NoError(t, err)
	defer writer.Close(ctx)
	held, err := writer.Begin(ctx)
	require.NoError(t, err)
	_, err = held.Exec(ctx, "LOCK TABLE "+embeddingTable+" IN ROW EXCLUSIVE MODE")
	require.NoError(t, err)

	require.Error(t, open(ctx), "init must time out while a writer holds the table")
	require.NoError(t, held.Rollback(ctx))

	bounded, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	require.NoError(t, open(bounded), "the failed init still holds the pool's only connection")
}
