package qdrant_test

import (
	"context"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/chains"
	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
	"github.com/vxcontrol/langchaingo/vectorstores/qdrant"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	tcqdrant "github.com/testcontainers/testcontainers-go/modules/qdrant"
)

func TestQdrantStore(t *testing.T) {
	t.Parallel()

	qdrantURL, apiKey, dimension, distance := getValues(t)
	collectionName := setupCollection(t, qdrantURL, apiKey, dimension, distance)
	opts := []openai.Option{
		openai.WithModel("gpt-4.1-nano"),
		openai.WithEmbeddingModel("text-embedding-ada-002"),
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	url, err := url.Parse(qdrantURL)
	require.NoError(t, err)
	store, err := qdrant.New(
		qdrant.WithURL(*url),
		qdrant.WithAPIKey(apiKey),
		qdrant.WithCollectionName(collectionName),
		qdrant.WithEmbedder(e),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(t.Context(), []schema.Document{
		{PageContent: "tokyo"},
		{PageContent: "potato"},
	})
	require.NoError(t, err)

	docs, err := store.SimilaritySearch(t.Context(), "japan", 1)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.Equal(t, "tokyo", docs[0].PageContent)
}

func TestQdrantStoreWithScoreThreshold(t *testing.T) {
	t.Parallel()

	qdrantURL, apiKey, dimension, distance := getValues(t)
	collectionName := setupCollection(t, qdrantURL, apiKey, dimension, distance)

	opts := []openai.Option{
		openai.WithModel("gpt-4.1-nano"),
		openai.WithEmbeddingModel("text-embedding-ada-002"),
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	url, err := url.Parse(qdrantURL)
	require.NoError(t, err)
	store, err := qdrant.New(
		qdrant.WithURL(*url),
		qdrant.WithAPIKey(apiKey),
		qdrant.WithCollectionName(collectionName),
		qdrant.WithEmbedder(e),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(t.Context(), []schema.Document{
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
	docs, err := store.SimilaritySearch(t.Context(),
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(0.8))
	require.NoError(t, err)
	require.Len(t, docs, 6)

	// test with a score threshold of 0, expected all 10 documents
	docs, err = store.SimilaritySearch(t.Context(),
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(0))
	require.NoError(t, err)
	require.Len(t, docs, 10)
}

func TestSimilaritySearchWithInvalidScoreThreshold(t *testing.T) {
	t.Parallel()

	qdrantURL, apiKey, dimension, distance := getValues(t)
	collectionName := setupCollection(t, qdrantURL, apiKey, dimension, distance)

	opts := []openai.Option{
		openai.WithModel("gpt-4.1-nano"),
		openai.WithEmbeddingModel("text-embedding-ada-002"),
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	url, err := url.Parse(qdrantURL)
	require.NoError(t, err)
	store, err := qdrant.New(
		qdrant.WithURL(*url),
		qdrant.WithAPIKey(apiKey),
		qdrant.WithCollectionName(collectionName),
		qdrant.WithEmbedder(e),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(t.Context(), []schema.Document{
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

	_, err = store.SimilaritySearch(t.Context(),
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(-0.8))
	require.Error(t, err)

	_, err = store.SimilaritySearch(t.Context(),
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(1.8))
	require.Error(t, err)
}

func TestQdrantAsRetriever(t *testing.T) {
	t.Parallel()

	qdrantURL, apiKey, dimension, distance := getValues(t)
	collectionName := setupCollection(t, qdrantURL, apiKey, dimension, distance)

	opts := []openai.Option{
		openai.WithModel("gpt-4.1-nano"),
		openai.WithEmbeddingModel("text-embedding-ada-002"),
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	url, err := url.Parse(qdrantURL)
	require.NoError(t, err)
	store, err := qdrant.New(
		qdrant.WithURL(*url),
		qdrant.WithAPIKey(apiKey),
		qdrant.WithCollectionName(collectionName),
		qdrant.WithEmbedder(e),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(
		t.Context(),
		[]schema.Document{
			{PageContent: "The color of the house is blue."},
			{PageContent: "The color of the car is red."},
			{PageContent: "The color of the desk is orange."},
		},
	)
	require.NoError(t, err)

	result, err := chains.Run(
		t.Context(),
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 1),
		),
		"What color is the desk?",
	)
	require.NoError(t, err)
	require.Contains(t, strings.ToLower(result), "orange", "expected orange in result")
}

func TestQdrantRetrieverScoreThreshold(t *testing.T) {
	t.Parallel()

	qdrantURL, apiKey, dimension, distance := getValues(t)
	collectionName := setupCollection(t, qdrantURL, apiKey, dimension, distance)

	opts := []openai.Option{
		openai.WithModel("gpt-4.1-nano"),
		openai.WithEmbeddingModel("text-embedding-ada-002"),
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	url, err := url.Parse(qdrantURL)
	require.NoError(t, err)
	store, err := qdrant.New(
		qdrant.WithURL(*url),
		qdrant.WithAPIKey(apiKey),
		qdrant.WithCollectionName(collectionName),
		qdrant.WithEmbedder(e),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(
		t.Context(),
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
		t.Context(),
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5, vectorstores.WithScoreThreshold(0.3)),
		),
		"What colors is each piece of furniture next to the desk?",
	)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(result), "black", "expected black in result")
	require.Contains(t, strings.ToLower(result), "beige", "expected beige in result")
}

func TestQdrantRetrieverFilter(t *testing.T) {
	t.Parallel()

	qdrantURL, apiKey, dimension, distance := getValues(t)
	collectionName := setupCollection(t, qdrantURL, apiKey, dimension, distance)

	opts := []openai.Option{
		openai.WithModel("gpt-4.1-nano"),
		openai.WithEmbeddingModel("text-embedding-ada-002"),
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	url, err := url.Parse(qdrantURL)
	require.NoError(t, err)
	store, err := qdrant.New(
		qdrant.WithURL(*url),
		qdrant.WithAPIKey(apiKey),
		qdrant.WithCollectionName(collectionName),
		qdrant.WithEmbedder(e),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(
		t.Context(),
		[]schema.Document{
			{PageContent: "The color of the house is blue."},
			{PageContent: "The color of the car is red."},
			{PageContent: "The color of the desk is orange."},
			{PageContent: "The color of the lamp beside the desk is black."},
			{PageContent: "The color of the chair beside the desk is beige."},
		},
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(
		t.Context(),
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
		},
	)
	require.NoError(t, err)

	filter := map[string]interface{}{
		"must": []map[string]interface{}{
			{
				"key": "location",
				"match": map[string]interface{}{
					"value": "office",
				},
			},
		},
	}

	result, err := chains.Run(
		t.Context(),
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5, vectorstores.WithFilters(filter)),
		),
		"What colors is the lamp?",
	)
	require.NoError(t, err)
	require.Contains(t, strings.ToLower(result), "orange", "expected orange in result")
}

func getValues(t *testing.T) (string, string, int, string) {
	t.Helper()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	qdrantURL := os.Getenv("QDRANT_URL")
	if qdrantURL == "" {
		qdrantContainer, err := tcqdrant.Run(t.Context(), "qdrant/qdrant:v1.7.4")
		if err != nil && strings.Contains(err.Error(), "Cannot connect to the Docker daemon") {
			t.Skip("Docker not available")
		}
		require.NoError(t, err)

		t.Cleanup(func() {
			ctx := context.Background() //nolint:usetesting
			require.NoError(t, qdrantContainer.Terminate(ctx))
		})

		qdrantURL, err = qdrantContainer.RESTEndpoint(t.Context())
		if err != nil {
			t.Skipf("Failed to get qdrant container endpoint: %s", err)
		}
	}

	// Can be empty if using a local Qdrant deployment
	apiKey := os.Getenv("QDRANT_API_KEY")

	// Reference: https://qdrant.tech/documentation/concepts/search/#metrics
	distance := os.Getenv("QDRANT_DISTANCE_METRIC")
	if distance == "" {
		distance = "Cosine"
	}
	embeddingDimension, err := strconv.Atoi(os.Getenv("QDRANT_EMBEDDING_DIMENSION"))
	if err != nil || embeddingDimension == 0 {
		embeddingDimension = 1536
	}
	return qdrantURL, apiKey, embeddingDimension, distance
}

func setupCollection(t *testing.T, qdrantURL, apiKey string, dimension int, distance string) string {
	t.Helper()

	collectionName := uuid.NewString()
	collectionConfig := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     dimension,
			"distance": distance,
		},
	}

	url, err := url.Parse(qdrantURL)
	require.NoError(t, err)

	url = url.JoinPath("collections", collectionName)
	_, status, err := qdrant.DoRequest(t.Context(), *url, apiKey, http.MethodPut, collectionConfig)

	require.Equal(t, http.StatusOK, status)
	require.NoError(t, err)

	t.Cleanup(func() {
		ctx := context.Background() //nolint:usetesting
		_, status, err := qdrant.DoRequest(ctx, *url, apiKey, http.MethodDelete, nil)

		require.Equal(t, http.StatusOK, status)
		require.NoError(t, err)
	})
	return collectionName
}
