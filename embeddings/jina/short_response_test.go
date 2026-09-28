package jina

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func embedAgainst(t *testing.T, body string, inputs ...string) ([][]float32, error) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)

	j, err := NewJina(WithAPIBaseURL(srv.URL), WithAPIKey("k"))
	require.NoError(t, err)
	return j.CreateEmbedding(context.Background(), inputs)
}

func TestAnEmptyEmbeddingListIsAFailure(t *testing.T) {
	t.Parallel()

	got, err := embedAgainst(t, `{"data":[]}`, "one")

	require.ErrorIs(t, err, ErrEmptyEmbeddings)
	assert.Nil(t, got, "a caller that gets an error must not also get a slice to index")
}

func TestFewerEmbeddingsThanInputsIsAFailure(t *testing.T) {
	t.Parallel()

	got, err := embedAgainst(t, `{"data":[{"embedding":[0.1,0.2]}]}`, "one", "two")

	require.ErrorIs(t, err, ErrShortEmbeddings)
	assert.Nil(t, got)
}

func TestOneEmbeddingPerInputIsTheHappyPath(t *testing.T) {
	t.Parallel()

	got, err := embedAgainst(t, `{"data":[{"embedding":[0.1,0.2]},{"embedding":[0.3,0.4]}]}`, "one", "two")

	require.NoError(t, err)
	require.Len(t, got, 2)
	assert.InDelta(t, float32(0.3), got[1][0], 1e-6)
}
