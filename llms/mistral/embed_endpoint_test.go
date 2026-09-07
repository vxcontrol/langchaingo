package mistral

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func embeddingPathOnTheWire(t *testing.T, prefix string) string {
	t.Helper()

	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.URL.Path
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2]}]}`))
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithEndpoint(srv.URL+prefix), WithAPIKey("k"))
	require.NoError(t, err)
	_, err = m.CreateEmbedding(context.Background(), []string{"x"})
	require.NoError(t, err)
	return got
}

func TestTheConfiguredEndpointKeepsItsPrefix(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "/v1/embeddings", embeddingPathOnTheWire(t, ""),
		"an endpoint that is a bare host still reaches the vendor's own path")
	assert.Equal(t, "/mistral/v1/embeddings", embeddingPathOnTheWire(t, "/mistral"),
		"a gateway prefix is part of the address the caller configured, not decoration")
}

func TestAnEmbeddingRefusalNamesWhatTheVendorSaid(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"message":"no route for that prefix"}`))
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithEndpoint(srv.URL), WithAPIKey("k"))
	require.NoError(t, err)

	_, err = m.CreateEmbedding(context.Background(), []string{"x"})
	require.ErrorIs(t, err, ErrEmbeddingFailed)
	assert.Contains(t, err.Error(), "no route for that prefix",
		"the caller sees what the vendor answered, not just that something failed")
}

func TestACancelledContextStopsTheEmbeddingCall(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1]}]}`))
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithEndpoint(srv.URL), WithAPIKey("k"))
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = m.CreateEmbedding(ctx, []string{"x"})
	require.ErrorIs(t, err, context.Canceled,
		"the caller's cancellation reaches the request, it is not ignored")
}
