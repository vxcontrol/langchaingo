package inmemory_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/chains"
	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
	"github.com/vxcontrol/langchaingo/vectorstores/inmemory"

	"github.com/stretchr/testify/require"
)

// mockEmbedder is a simple embedder that returns predictable embeddings for testing
type mockEmbedder struct{}

func (m *mockEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		switch text {
		case "similar1":
			// very similar to query "similar"
			embeddings[i] = []float32{1.0, 0.9, 0.8}
		case "similar2":
			// similar to query "similar"
			embeddings[i] = []float32{0.9, 0.8, 0.7}
		case "different":
			// different from query "similar"
			embeddings[i] = []float32{0.1, 0.2, 0.3}
		default:
			// default embedding
			embeddings[i] = []float32{0.0, 0.0, 0.0}
		}
	}
	return embeddings, nil
}

func (m *mockEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	if text == "similar" {
		return []float32{1.0, 0.9, 0.8}, nil
	}
	return []float32{0.0, 0.0, 0.0}, nil
}

func TestMockSimilarityScoreCalculation(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mockEmb := &mockEmbedder{}

	store, err := inmemory.New(
		ctx,
		inmemory.WithEmbedder(mockEmb),
		inmemory.WithVectorSize(3), // our mock embeddings are 3-dimensional
	)
	require.NoError(t, err)

	// add documents with different similarities to our query "similar"
	_, err = store.AddDocuments(ctx, []schema.Document{
		{PageContent: "similar1"},  // should have highest similarity (almost identical to query)
		{PageContent: "similar2"},  // should have high similarity
		{PageContent: "different"}, // should have low similarity
	})
	require.NoError(t, err)

	// search with query "similar"
	docs, err := store.SimilaritySearch(ctx, "similar", 3)
	require.NoError(t, err)

	// print results for debugging
	t.Logf("Search results: %+v", docs)

	// test expects documents to be returned
	require.GreaterOrEqual(t, len(docs), 1, "at least one document should be returned")

	// find document with highest score
	var highestScoreDoc schema.Document
	var highestScore float32 = -1

	for _, doc := range docs {
		if doc.Score > highestScore {
			highestScore = doc.Score
			highestScoreDoc = doc
		}
	}

	// check that the document with highest score is similar1 or similar2
	require.Contains(t, []string{"similar1", "similar2"}, highestScoreDoc.PageContent,
		"Document with highest score should be one of the similar documents")

	// test with score threshold
	docsFiltered, err := store.SimilaritySearch(ctx, "similar", 3, vectorstores.WithScoreThreshold(0.8))
	require.NoError(t, err)

	// print filtered results for debugging
	t.Logf("Filtered results: %+v", docsFiltered)

	// check that filtered results meet threshold
	for _, doc := range docsFiltered {
		require.GreaterOrEqual(t, doc.Score, float32(0.8), "filtered documents should all have scores >= threshold")
	}
}

func preCheckEnvSetting(t *testing.T) {
	t.Helper()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
}

func TestInMemoryStoreRest(t *testing.T) {
	t.Parallel()
	preCheckEnvSetting(t)
	ctx := context.Background()

	llm, err := openai.New()
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	store, err := inmemory.New(
		ctx,
		inmemory.WithEmbedder(e),
		inmemory.WithVectorSize(1536),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(ctx, []schema.Document{
		{PageContent: "tokyo", Metadata: map[string]any{
			"country": "japan",
		}},
		{PageContent: "potato"},
	})
	require.NoError(t, err)

	docs, err := store.SimilaritySearch(ctx, "japan", 1)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.Equal(t, "tokyo", docs[0].PageContent)
	require.Equal(t, "japan", docs[0].Metadata["country"])
}

func TestInMemoryStoreRestWithScoreThreshold(t *testing.T) {
	t.Parallel()
	preCheckEnvSetting(t)
	ctx := context.Background()

	llm, err := openai.New()
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	store, err := inmemory.New(
		ctx,
		inmemory.WithEmbedder(e),
		inmemory.WithVectorSize(1536),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(context.Background(), []schema.Document{
		{PageContent: "Tokyo"},
		{PageContent: "Yokohama"},
		{PageContent: "Osaka"},
		{PageContent: "Nagoya"},
		{PageContent: "Sapporo"},
		{PageContent: "Fukuoka"},
		{PageContent: "Dublin"},
		{PageContent: "Paris"},
		{PageContent: "London "},
		{PageContent: "New York"},
	})
	require.NoError(t, err)

	// test with a score threshold of 0.8, expected 6 documents
	docs, err := store.SimilaritySearch(
		ctx,
		"Which of these are cities in Japan",
		10,
		vectorstores.WithScoreThreshold(0.8),
	)
	require.NoError(t, err)
	require.Len(t, docs, 6)

	// test with a score threshold of 0, expected all 10 documents
	docs, err = store.SimilaritySearch(
		ctx,
		"Which of these are cities in Japan",
		10,
		vectorstores.WithScoreThreshold(0))
	require.NoError(t, err)
	require.Len(t, docs, 10)
}

func TestSimilaritySearchWithInvalidScoreThreshold(t *testing.T) {
	t.Parallel()
	preCheckEnvSetting(t)
	ctx := context.Background()

	llm, err := openai.New()
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	store, err := inmemory.New(
		ctx,
		inmemory.WithEmbedder(e),
		inmemory.WithVectorSize(1536),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(ctx, []schema.Document{
		{PageContent: "Tokyo"},
		{PageContent: "Yokohama"},
		{PageContent: "Osaka"},
		{PageContent: "Nagoya"},
		{PageContent: "Sapporo"},
		{PageContent: "Fukuoka"},
		{PageContent: "Dublin"},
		{PageContent: "Paris"},
		{PageContent: "London "},
		{PageContent: "New York"},
	})
	require.NoError(t, err)

	_, err = store.SimilaritySearch(
		ctx,
		"Which of these are cities in Japan",
		10,
		vectorstores.WithScoreThreshold(-0.8),
	)
	require.Error(t, err)

	_, err = store.SimilaritySearch(
		ctx,
		"Which of these are cities in Japan",
		10,
		vectorstores.WithScoreThreshold(1.8),
	)
	require.Error(t, err)
}

func TestInMemoryAsRetriever(t *testing.T) {
	t.Parallel()
	preCheckEnvSetting(t)
	ctx := context.Background()

	llm, err := openai.New()
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	store, err := inmemory.New(
		ctx,
		inmemory.WithEmbedder(e),
		inmemory.WithVectorSize(1536),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(
		ctx,
		[]schema.Document{
			{PageContent: "The color of the house is blue."},
			{PageContent: "The color of the car is red."},
			{PageContent: "The color of the desk is orange."},
		},
	)
	require.NoError(t, err)

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 1),
		),
		"What color is the desk?",
	)
	require.NoError(t, err)
	require.True(t, strings.Contains(result, "orange"), "expected orange in result")
}

func TestInMemoryAsRetrieverWithScoreThreshold(t *testing.T) {
	t.Parallel()
	preCheckEnvSetting(t)
	ctx := context.Background()

	llm, err := openai.New()
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	store, err := inmemory.New(
		ctx,
		inmemory.WithEmbedder(e),
		inmemory.WithVectorSize(1536),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(
		context.Background(),
		[]schema.Document{
			{PageContent: "The color of the house is blue."},
			{PageContent: "The color of the car is red."},
			{PageContent: "The color of the desk is orange."},
			{PageContent: "The color of the lamp beside the desk is black."},
			{PageContent: "The color of the chair beside the desk is beige."},
		},
	)
	require.NoError(t, err)

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5, vectorstores.WithScoreThreshold(0.8)),
		),
		"What colors are all of the pieces of furniture next to the desk and the desk itself?",
	)
	require.NoError(t, err)

	require.Contains(t, result, "orange", "expected orange in result")
	require.Contains(t, result, "black", "expected black in result")
	require.Contains(t, result, "beige", "expected beige in result")
}

func TestInMemoryAsRetrieverWithMetadataFilterNotSelected(t *testing.T) {
	t.Parallel()
	preCheckEnvSetting(t)
	ctx := context.Background()

	llm, err := openai.New()
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	store, err := inmemory.New(
		ctx,
		inmemory.WithEmbedder(e),
		inmemory.WithVectorSize(1536),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(
		ctx,
		[]schema.Document{
			{
				PageContent: "The color of the lamp beside the desk is black.",
				Metadata: map[string]any{
					"location": "kitchen",
				},
			},
			{
				PageContent: "The color of the lamp beside the desk is blue.",
				Metadata: map[string]any{
					"location": "bedroom",
				},
			},
			{
				PageContent: "The color of the lamp beside the desk is orange.",
				Metadata: map[string]any{
					"location": "office",
				},
			},
			{
				PageContent: "The color of the lamp beside the desk is purple.",
				Metadata: map[string]any{
					"location": "sitting room",
				},
			},
			{
				PageContent: "The color of the lamp beside the desk is yellow.",
				Metadata: map[string]any{
					"location": "patio",
				},
			},
		},
	)
	require.NoError(t, err)

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5),
		),
		"What color is the lamp in each room?",
	)
	require.NoError(t, err)

	require.Contains(t, result, "black", "expected black in result")
	require.Contains(t, result, "blue", "expected blue in result")
	require.Contains(t, result, "orange", "expected orange in result")
	require.Contains(t, result, "purple", "expected purple in result")
	require.Contains(t, result, "yellow", "expected yellow in result")
}

func TestInMemoryAsRetrieverWithMetadataFilters(t *testing.T) {
	t.Parallel()
	preCheckEnvSetting(t)
	ctx := context.Background()

	llm, err := openai.New()
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	store, err := inmemory.New(
		ctx,
		inmemory.WithEmbedder(e),
		inmemory.WithVectorSize(1536),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(
		context.Background(),
		[]schema.Document{
			{
				PageContent: "The color of the lamp beside the desk is orange.",
				Metadata: map[string]any{
					"location":    "office",
					"square_feet": 100,
				},
			},
			{
				PageContent: "The color of the lamp beside the desk is purple.",
				Metadata: map[string]any{
					"location":    "sitting room",
					"square_feet": 400,
				},
			},
			{
				PageContent: "The color of the lamp beside the desk is yellow.",
				Metadata: map[string]any{
					"location":    "patio",
					"square_feet": 800,
				},
			},
		},
	)
	require.NoError(t, err)

	filter := map[string]any{"location": "sitting room"}

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store,
				5,
				vectorstores.WithFilters(filter))),
		"What color is the lamp in each room?",
	)
	require.NoError(t, err)
	require.Contains(t, result, "purple", "expected purple in result")
	require.NotContains(t, result, "orange", "expected not orange in result")
	require.NotContains(t, result, "yellow", "expected not yellow in result")
}

func TestDeduplicater(t *testing.T) {
	t.Parallel()
	preCheckEnvSetting(t)
	ctx := context.Background()

	llm, err := openai.New()
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	store, err := inmemory.New(
		ctx,
		inmemory.WithEmbedder(e),
		inmemory.WithVectorSize(1536),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(context.Background(), []schema.Document{
		{PageContent: "tokyo", Metadata: map[string]any{
			"type": "city",
		}},
		{PageContent: "potato", Metadata: map[string]any{
			"type": "vegetable",
		}},
	}, vectorstores.WithDeduplicater(
		func(ctx context.Context, doc schema.Document) bool {
			return doc.PageContent == "tokyo"
		},
	))
	require.NoError(t, err)

	docs, err := store.SimilaritySearch(ctx, "potato", 1)
	require.NoError(t, err)

	require.Len(t, docs, 1)
	require.Equal(t, "potato", docs[0].PageContent)
	require.Equal(t, "vegetable", docs[0].Metadata["type"])
}
