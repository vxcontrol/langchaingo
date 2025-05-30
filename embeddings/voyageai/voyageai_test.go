package voyageai

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVoyageAIEmbeddings(t *testing.T) {
	t.Parallel()

	if voyageaiKey := os.Getenv("VOYAGEAI_API_KEY"); voyageaiKey == "" {
		t.Skip("VOYAGEAI_API_KEY not set")
	}

	e, err := NewVoyageAI()
	require.NoError(t, err)

	_, err = e.EmbedQuery(t.Context(), "Hello world!")
	require.NoError(t, err)

	embeddings, err := e.EmbedDocuments(t.Context(), []string{
		"Hello world",
		"The world is ending",
		"good bye",
	})
	require.NoError(t, err)
	assert.Len(t, embeddings, 3)
}
