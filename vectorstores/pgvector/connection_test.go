package pgvector

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func withApplicationName(t *testing.T, url string) (string, string) {
	t.Helper()

	name := "pgvector_" + strings.ReplaceAll(uuid.New().String(), "-", "")
	separator := "?"
	if strings.Contains(url, "?") {
		separator = "&"
	}
	return url + separator + "application_name=" + name, name
}

func openConnectionsNamed(t *testing.T, url, name string) func() int {
	t.Helper()

	conn, err := pgx.Connect(t.Context(), url)
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close(context.Background()) })
	return func() int {
		var count int
		require.NoError(t, conn.QueryRow(t.Context(),
			"SELECT count(*) FROM pg_stat_activity WHERE application_name = $1", name).Scan(&count))
		return count
	}
}

func TestClosingAStoreClosesTheConnectionItOpened(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	named, name := withApplicationName(t, url)
	count := openConnectionsNamed(t, url, name)

	store, err := New(t.Context(), WithConnectionURL(named), WithEmbedder(fixedEmbedder{dims: 64}),
		WithCollectionName(narrowingCollection()))
	require.NoError(t, err)
	require.Equal(t, 1, count())

	require.NoError(t, store.Close())
	assert.Eventually(t, func() bool { return count() == 0 }, 5*time.Second, 50*time.Millisecond)
}

func TestAFailedNewClosesTheConnectionItOpened(t *testing.T) {
	t.Parallel()

	url := narrowingURL(t)
	named, name := withApplicationName(t, url)
	count := openConnectionsNamed(t, url, name)

	_, err := New(t.Context(), WithConnectionURL(named), WithEmbedder(fixedEmbedder{dims: 64}),
		WithCollectionName(narrowingCollection()), WithEmbeddingTableName("not a table name"))
	require.Error(t, err)

	assert.Eventually(t, func() bool { return count() == 0 }, 5*time.Second, 50*time.Millisecond)
}
