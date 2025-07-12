package googleai

import (
	"context"
	"fmt"

	"google.golang.org/genai"
)

// CreateEmbedding creates embeddings from texts.
func (g *GoogleAI) CreateEmbedding(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	results := make([][]float32, 0, len(texts))

	// Process texts in batches of 100 as per API limits
	for i := 0; i < len(texts); i += 100 {
		end := i + 100
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]
		batchResults, err := g.processEmbeddingBatch(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("failed to create embeddings for batch starting at %d: %w", i, err)
		}

		results = append(results, batchResults...)
	}

	return results, nil
}

func (g *GoogleAI) processEmbeddingBatch(ctx context.Context, texts []string) ([][]float32, error) {
	contents := make([]*genai.Content, 0, len(texts))

	for _, text := range texts {
		content := genai.NewContentFromText(text, genai.RoleUser)
		contents = append(contents, content)
	}

	response, err := g.client.Models.EmbedContent(ctx, g.opts.DefaultEmbeddingModel, contents, nil)
	if err != nil {
		return nil, err
	}

	results := make([][]float32, 0, len(response.Embeddings))
	for _, embedding := range response.Embeddings {
		results = append(results, embedding.Values)
	}

	return results, nil
}
