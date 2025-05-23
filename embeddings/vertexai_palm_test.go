package embeddings

import (
	"os"
	"testing"

	"github.com/vxcontrol/langchaingo/llms/googleai/palm"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newVertexEmbedder(t *testing.T, opts ...Option) *EmbedderImpl {
	t.Helper()

	if gcpProjectID := os.Getenv("GOOGLE_CLOUD_PROJECT"); gcpProjectID == "" {
		t.Skip("GOOGLE_CLOUD_PROJECT not set")
		return nil
	}

	llm, err := palm.New()
	require.NoError(t, err)

	embedder, err := NewEmbedder(llm, opts...)
	require.NoError(t, err)

	return embedder
}

func TestVertexAIPaLMEmbeddings(t *testing.T) {
	t.Parallel()

	e := newVertexEmbedder(t)

	_, err := e.EmbedQuery(t.Context(), "Hello world!")
	require.NoError(t, err)

	embeddings, err := e.EmbedDocuments(t.Context(), []string{
		"Hello world",
		"The world is ending",
		"good bye",
	})
	require.NoError(t, err)
	assert.Len(t, embeddings, 3)
}

func TestVertexAIPaLMEmbeddingsWithOptions(t *testing.T) {
	t.Parallel()

	e := newVertexEmbedder(t, WithBatchSize(5), WithStripNewLines(false))

	_, err := e.EmbedQuery(t.Context(), "Hello world!")
	require.NoError(t, err)

	embeddings, err := e.EmbedDocuments(t.Context(), []string{"Hello world"})
	require.NoError(t, err)
	assert.Len(t, embeddings, 1)
}
