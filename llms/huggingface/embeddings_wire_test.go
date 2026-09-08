package huggingface

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func embeddingsPathOf(t *testing.T, opts []Option, model, task string) string {
	t.Helper()

	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[[0.1,0.2]]`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(append([]Option{WithToken("t"), WithURL(srv.URL)}, opts...)...)
	require.NoError(t, err)

	_, err = llm.CreateEmbedding(context.Background(), []string{"hi"}, model, task)
	require.NoError(t, err)

	return path
}

func TestTheEmbeddingsPathCarriesTheProviderAndTheTask(t *testing.T) {
	t.Parallel()

	path := embeddingsPathOf(t, nil, "BAAI/bge-small-en-v1.5", "feature-extraction")

	assert.Equal(t, "/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction", path)
}

func TestTheNamedInferenceProviderReplacesTheDefaultSegment(t *testing.T) {
	t.Parallel()

	path := embeddingsPathOf(t, []Option{WithInferenceProvider("scaleway")},
		"Qwen/Qwen3-Embedding-8B", "feature-extraction")

	assert.Equal(t, "/scaleway/models/Qwen/Qwen3-Embedding-8B/pipeline/feature-extraction", path)
}
