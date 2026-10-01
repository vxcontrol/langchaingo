package pgvector

import (
	"context"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func withParameter(t *testing.T, connString, key, value string) string {
	t.Helper()

	parsed, err := url.Parse(connString)
	require.NoError(t, err)
	query := parsed.Query()
	query.Set(key, value)
	parsed.RawQuery = query.Encode()
	return parsed.String()
}

func openConnectionsNamed(t *testing.T, connString, name string) func() int {
	t.Helper()

	conn, err := pgx.Connect(t.Context(), connString)
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close(context.Background()) })
	return func() int {
		var count int
		require.NoError(t, conn.QueryRow(t.Context(),
			"SELECT count(*) FROM pg_stat_activity WHERE application_name = $1", name).Scan(&count))
		return count
	}
}

func applicationName() string {
	return "pgvector_" + strings.ReplaceAll(uuid.New().String(), "-", "")
}

func TestClosingAStoreClosesTheConnectionItOpened(t *testing.T) {
	t.Parallel()

	connString := narrowingURL(t)
	name := applicationName()
	count := openConnectionsNamed(t, connString, name)

	store, err := New(t.Context(), WithConnectionURL(withParameter(t, connString, "application_name", name)),
		WithEmbedder(fixedEmbedder{dims: 64}), WithCollectionName(narrowingCollection()))
	require.NoError(t, err)
	require.Equal(t, 1, count())

	require.NoError(t, store.Close())
	assert.Eventually(t, func() bool { return count() == 0 }, 5*time.Second, 50*time.Millisecond)
}

func TestAFailedNewClosesTheConnectionItOpened(t *testing.T) {
	t.Parallel()

	connString := narrowingURL(t)
	name := applicationName()
	count := openConnectionsNamed(t, connString, name)

	_, err := New(t.Context(), WithConnectionURL(withParameter(t, connString, "application_name", name)),
		WithEmbedder(fixedEmbedder{dims: 64}), WithCollectionName(narrowingCollection()),
		WithEmbeddingTableName("not a table name"))
	var pgErr *pgconn.PgError
	require.ErrorAs(t, err, &pgErr)
	require.Equal(t, "42601", pgErr.Code)

	assert.Eventually(t, func() bool { return count() == 0 }, 5*time.Second, 50*time.Millisecond)
}

func TestClosingAStoreLeavesTheCallersConnectionOpen(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	conn, err := pgx.Connect(ctx, narrowingURL(t))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close(context.Background()) })

	store, err := New(ctx, WithConn(conn), WithEmbedder(fixedEmbedder{dims: 64}),
		WithCollectionName(narrowingCollection()))
	require.NoError(t, err)
	require.NoError(t, store.Close())

	require.NoError(t, conn.Ping(ctx))
}
