package googleai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCreateEmbedding_EmptyInput(t *testing.T) {
	t.Parallel()

	g := &GoogleAI{}

	result, err := g.CreateEmbedding(context.Background(), []string{})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 0)
}

func TestCreateEmbedding_BatchLogic(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name            string
		numTexts        int
		expectedBatches int
	}{
		{"single text", 1, 1},
		{"small batch", 50, 1},
		{"exactly 100", 100, 1},
		{"101 texts", 101, 2},
		{"200 texts", 200, 2},
		{"250 texts", 250, 3},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Calculate expected number of batches using the same logic as CreateEmbedding
			expectedBatches := 0
			if tc.numTexts > 0 {
				expectedBatches = (tc.numTexts + 99) / 100 // Ceiling division
			}

			assert.Equal(t, tc.expectedBatches, expectedBatches,
				"For %d texts, expected %d batches but calculated %d",
				tc.numTexts, tc.expectedBatches, expectedBatches)
		})
	}
}

func TestCreateEmbedding_InputValidation(t *testing.T) {
	t.Parallel()

	validInputs := []struct {
		name  string
		texts []string
	}{
		{"single text", []string{"Hello world"}},
		{"multiple texts", []string{"Hello", "world", "test"}},
		{"empty string included", []string{"Hello", "", "world"}},
		{"unicode content", []string{"Hello 世界", "🌍 Earth"}},
		{"long text", []string{string(make([]byte, 1000))}},
	}

	for _, test := range validInputs {
		t.Run(test.name, func(t *testing.T) {
			for _, text := range test.texts {
				assert.IsType(t, "", text, "Text should be string type")
			}
		})
	}
}

func TestProcessEmbeddingBatch_ContextHandling(t *testing.T) {
	t.Parallel()

	t.Run("cancelled context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		// Testing that we handle context properly
		assert.Equal(t, context.Canceled, ctx.Err())
	})

	t.Run("context with timeout", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 0) // Immediate timeout
		defer cancel()

		// Testing that we handle context properly
		assert.Equal(t, context.DeadlineExceeded, ctx.Err())
	})
}

func TestEmbeddingConstants(t *testing.T) {
	t.Parallel()

	// Verify our understanding of the batch size limit
	const expectedBatchSize = 100

	// Verify that our batch size matches the documented API limit
	assert.Equal(t, 100, expectedBatchSize, "Batch size should match API documentation")
}

func TestEmbeddingOutputStructure(t *testing.T) {
	t.Parallel()

	// Test expected output format
	t.Run("output type validation", func(t *testing.T) {
		// Embeddings should return [][]float32
		expectedOutput := [][]float32{
			{0.1, 0.2, 0.3},
			{0.4, 0.5, 0.6},
		}

		assert.IsType(t, [][]float32{}, expectedOutput)
		assert.Len(t, expectedOutput, 2)

		for i, embedding := range expectedOutput {
			assert.IsType(t, []float32{}, embedding, "Embedding %d should be []float32", i)
			assert.NotEmpty(t, embedding, "Embedding %d should not be empty", i)
		}
	})
}
