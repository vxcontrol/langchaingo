package embeddings

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBatchTexts(t *testing.T) {
	t.Parallel()

	cases := []struct {
		texts     []string
		batchSize int
		expected  [][]string
	}{
		{
			texts:     []string{},
			batchSize: 1,
			expected:  [][]string{},
		},
		{
			texts:     []string{"foo bar zoo"},
			batchSize: 4,
			expected:  [][]string{{"foo bar zoo"}},
		},
		{
			texts:     []string{"foo bar zoo", "foo"},
			batchSize: 7,
			expected:  [][]string{{"foo bar zoo", "foo"}},
		},
		{
			texts:     []string{"foo", "bar", "zoo"},
			batchSize: 2,
			expected:  [][]string{{"foo", "bar"}, {"zoo"}},
		},
		{
			texts:     []string{"foo", "bar", "zoo", "baz", "qux"},
			batchSize: 2,
			expected:  [][]string{{"foo", "bar"}, {"zoo", "baz"}, {"qux"}},
		},
		{
			texts:     []string{"foo", "bar", "zoo", "baz"},
			batchSize: 2,
			expected:  [][]string{{"foo", "bar"}, {"zoo", "baz"}},
		},
		{
			texts:     []string{"foo", "bar", "zoo", "baz", "qux"},
			batchSize: 3,
			expected:  [][]string{{"foo", "bar", "zoo"}, {"baz", "qux"}},
		},
		{
			texts:     []string{"foo", "bar", "zoo", "baz", "qux"},
			batchSize: 6,
			expected:  [][]string{{"foo", "bar", "zoo", "baz", "qux"}},
		},
	}

	for _, tc := range cases {
		assert.Equal(t, tc.expected, BatchTexts(tc.texts, tc.batchSize))
	}
}

func TestEmbedQueryRefusesAnAnswerWithoutAVector(t *testing.T) {
	t.Parallel()

	client := EmbedderClientFunc(func(context.Context, []string) ([][]float32, error) {
		return [][]float32{}, nil
	})
	e, err := NewEmbedder(client)
	require.NoError(t, err)

	got, err := e.EmbedQuery(context.Background(), "hello")

	require.ErrorIs(t, err, ErrNoEmbedding)
	assert.Nil(t, got)
}
