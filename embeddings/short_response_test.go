package embeddings

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBatchedEmbedRefusesAnAnswerWithoutVectors(t *testing.T) {
	t.Parallel()

	client := EmbedderClientFunc(func(context.Context, []string) ([][]float32, error) {
		return [][]float32{}, nil
	})

	emb, err := BatchedEmbed(t.Context(), client, []string{"one"}, 8)
	require.ErrorIs(t, err, ErrNoEmbedding)
	require.Nil(t, emb)
}

func TestBatchedEmbedRefusesAShortBatch(t *testing.T) {
	t.Parallel()

	client := EmbedderClientFunc(func(_ context.Context, texts []string) ([][]float32, error) {
		return [][]float32{{0.1}}, nil
	})

	emb, err := BatchedEmbed(t.Context(), client, []string{"one", "two"}, 8)
	require.ErrorIs(t, err, ErrShortEmbedding)
	require.Nil(t, emb)
}

func TestBatchedEmbedKeepsEveryVectorOfAFullAnswer(t *testing.T) {
	t.Parallel()

	client := EmbedderClientFunc(func(_ context.Context, texts []string) ([][]float32, error) {
		out := make([][]float32, 0, len(texts))
		for range texts {
			out = append(out, []float32{0.1})
		}
		return out, nil
	})

	emb, err := BatchedEmbed(t.Context(), client, []string{"one", "two", "three"}, 2)
	require.NoError(t, err)
	require.Len(t, emb, 3)
}
