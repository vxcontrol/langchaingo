package pinecone_test

import (
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vxcontrol/langchaingo/chains"
	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
	"github.com/vxcontrol/langchaingo/vectorstores/pinecone"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

// getValues returns Pinecone API credentials for testing.
//
// ARCHITECTURAL NOTE: Pinecone tests skip when credentials are not available
// instead of using httprr because the Pinecone client does not support custom
// HTTP clients, making it impossible to use httprr for HTTP mocking.
// This is a legitimate architectural exception to the standard httprr pattern.
func getValues(t *testing.T) (string, string) {
	t.Helper()

	// Skip test if credentials are not available - Pinecone tests require real credentials
	// since Pinecone client doesn't support custom HTTP clients for httprr mocking
	pineconeAPIKey := os.Getenv("PINECONE_API_KEY")
	pineconeHost := os.Getenv("PINECONE_HOST")
	if pineconeAPIKey == "" || pineconeHost == "" {
		t.Skip("Pinecone tests require PINECONE_API_KEY and PINECONE_HOST environment variables")
	}

	return pineconeAPIKey, pineconeHost
}

func waitPineconeIndexing(t *testing.T) {
	t.Helper()

	// small delay for pinecone to index the documents
	// it's a dirty hack, but sometimes the tests fail because of the pinecone indexing
	select {
	case <-time.After(12 * time.Second):
	case <-t.Context().Done():
		t.Fatal("test timed out")
	}
}

// createOpenAIEmbedder creates an OpenAI embedder with httprr support for testing.
func createOpenAIEmbedder(t *testing.T, rr *httprr.RecordReplay) *embeddings.EmbedderImpl {
	t.Helper()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")

	opts := []openai.Option{
		openai.WithEmbeddingModel("text-embedding-ada-002"),
		openai.WithHTTPClient(rr.Client()),
	}
	if !rr.Recording() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)
	return e
}

// createOpenAILLMAndEmbedder creates both LLM and embedder with httprr support for chain tests.
func createOpenAILLMAndEmbedder(t *testing.T, rr *httprr.RecordReplay) (*openai.LLM, *embeddings.EmbedderImpl) {
	t.Helper()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")

	opts := []openai.Option{
		openai.WithHTTPClient(rr.Client()),
	}
	embeddingOpts := []openai.Option{
		openai.WithEmbeddingModel("text-embedding-ada-002"),
		openai.WithHTTPClient(rr.Client()),
	}
	if !rr.Recording() {
		opts = append(opts, openai.WithToken("test-api-key"))
		embeddingOpts = append(embeddingOpts, openai.WithToken("test-api-key"))
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)
	embeddingLLM, err := openai.New(embeddingOpts...)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(embeddingLLM)
	require.NoError(t, err)
	return llm, e
}

func TestPineconeStoreRest(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	apiKey, host := getValues(t)
	e := createOpenAIEmbedder(t, rr)

	storer, err := pinecone.New(
		pinecone.WithAPIKey(apiKey),
		pinecone.WithHost(host),
		pinecone.WithEmbedder(e),
		pinecone.WithNameSpace(uuid.New().String()),
	)
	require.NoError(t, err)

	_, err = storer.AddDocuments(ctx, []schema.Document{
		{PageContent: "tokyo"},
		{PageContent: "potato"},
	})
	require.NoError(t, err)

	waitPineconeIndexing(t)

	docs, err := storer.SimilaritySearch(ctx, "japan", 1)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.Equal(t, "tokyo", docs[0].PageContent)
}

func TestPineconeStoreRestWithScoreThreshold(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	apiKey, host := getValues(t)
	e := createOpenAIEmbedder(t, rr)

	storer, err := pinecone.New(
		pinecone.WithAPIKey(apiKey),
		pinecone.WithHost(host),
		pinecone.WithEmbedder(e),
		pinecone.WithNameSpace(uuid.New().String()),
	)
	require.NoError(t, err)

	_, err = storer.AddDocuments(ctx, []schema.Document{
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

	waitPineconeIndexing(t)

	// test with a score threshold of 0.8, expected 6 documents
	docs, err := storer.SimilaritySearch(ctx,
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(0.8))
	require.NoError(t, err)
	require.Len(t, docs, 6)

	// test with a score threshold of 0, expected all 10 documents
	docs, err = storer.SimilaritySearch(ctx,
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(0))
	require.NoError(t, err)
	require.Len(t, docs, 10)
}

func TestSimilaritySearchWithInvalidScoreThreshold(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	apiKey, host := getValues(t)
	e := createOpenAIEmbedder(t, rr)

	storer, err := pinecone.New(
		pinecone.WithAPIKey(apiKey),
		pinecone.WithHost(host),
		pinecone.WithEmbedder(e),
		pinecone.WithNameSpace(uuid.New().String()),
	)
	require.NoError(t, err)

	_, err = storer.AddDocuments(ctx, []schema.Document{
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

	waitPineconeIndexing(t)

	_, err = storer.SimilaritySearch(ctx,
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(-0.8))
	require.Error(t, err)

	_, err = storer.SimilaritySearch(ctx,
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(1.8))
	require.Error(t, err)
}

func TestPineconeAsRetriever(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	apiKey, host := getValues(t)
	llm, e := createOpenAILLMAndEmbedder(t, rr)

	store, err := pinecone.New(
		pinecone.WithAPIKey(apiKey),
		pinecone.WithHost(host),
		pinecone.WithEmbedder(e),
	)
	require.NoError(t, err)

	id := uuid.New().String()

	_, err = store.AddDocuments(
		ctx,
		[]schema.Document{
			{PageContent: "The color of the house is blue."},
			{PageContent: "The color of the car is red."},
			{PageContent: "The color of the desk is orange."},
		},
		vectorstores.WithNameSpace(id),
	)
	require.NoError(t, err)

	waitPineconeIndexing(t)

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 1, vectorstores.WithNameSpace(id)),
		),
		"What color is the desk?",
	)
	require.NoError(t, err)
	require.Contains(t, strings.ToLower(result), "orange", "expected orange in result")
}

func TestPineconeAsRetrieverWithScoreThreshold(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	apiKey, host := getValues(t)
	llm, e := createOpenAILLMAndEmbedder(t, rr)

	store, err := pinecone.New(
		pinecone.WithAPIKey(apiKey),
		pinecone.WithHost(host),
		pinecone.WithEmbedder(e),
	)
	require.NoError(t, err)

	id := uuid.New().String()

	_, err = store.AddDocuments(
		ctx,
		[]schema.Document{
			{PageContent: "The color of the house is blue."},
			{PageContent: "The color of the car is red."},
			{PageContent: "The color of the desk is orange."},
			{PageContent: "The color of the lamp beside the desk is black."},
			{PageContent: "The color of the chair beside the desk is beige."},
		},
		vectorstores.WithNameSpace(id),
	)
	require.NoError(t, err)

	waitPineconeIndexing(t)

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5, vectorstores.WithNameSpace(
				id), vectorstores.WithScoreThreshold(0.8)),
		),
		"What colors is each piece of furniture next to the desk?",
	)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(result), "black", "expected black in result")
	require.Contains(t, strings.ToLower(result), "beige", "expected beige in result")
}

func TestPineconeAsRetrieverWithMetadataFilterEqualsClause(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	apiKey, host := getValues(t)
	llm, e := createOpenAILLMAndEmbedder(t, rr)

	store, err := pinecone.New(
		pinecone.WithAPIKey(apiKey),
		pinecone.WithHost(host),
		pinecone.WithEmbedder(e),
	)
	require.NoError(t, err)

	id := uuid.New().String()

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
		vectorstores.WithNameSpace(id),
	)
	require.NoError(t, err)

	waitPineconeIndexing(t)

	filter := make(map[string]any)
	filterValue := make(map[string]any)
	filterValue["$eq"] = "patio"
	filter["location"] = filterValue

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5, vectorstores.WithNameSpace(
				id), vectorstores.WithFilters(filter)),
		),
		"What colors is the lamp?",
	)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(result), "yellow", "expected yellow in result")
}

func TestPineconeAsRetrieverWithMetadataFilterInClause(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	apiKey, host := getValues(t)
	llm, e := createOpenAILLMAndEmbedder(t, rr)

	store, err := pinecone.New(
		pinecone.WithAPIKey(apiKey),
		pinecone.WithHost(host),
		pinecone.WithEmbedder(e),
	)
	require.NoError(t, err)

	id := uuid.New().String()

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
		vectorstores.WithNameSpace(id),
	)
	require.NoError(t, err)

	waitPineconeIndexing(t)

	filter := make(map[string]any)
	filterValue := make(map[string]any)
	filterValue["$in"] = []string{"office", "kitchen"}
	filter["location"] = filterValue

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5, vectorstores.WithNameSpace(
				id), vectorstores.WithFilters(filter)),
		),
		"What color is the lamp in each room?",
	)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(result), "black", "expected black in result")
	require.Contains(t, strings.ToLower(result), "orange", "expected orange in result")
}

func TestPineconeAsRetrieverWithMetadataFilterNotSelected(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	apiKey, host := getValues(t)
	llm, e := createOpenAILLMAndEmbedder(t, rr)

	store, err := pinecone.New(
		pinecone.WithAPIKey(apiKey),
		pinecone.WithHost(host),
		pinecone.WithEmbedder(e),
	)
	require.NoError(t, err)

	id := uuid.New().String()

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
		vectorstores.WithNameSpace(id),
	)
	require.NoError(t, err)

	waitPineconeIndexing(t)

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5, vectorstores.WithNameSpace(
				id)),
		),
		"What color is the lamp in each room?",
	)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(result), "black", "expected black in result")
	require.Contains(t, strings.ToLower(result), "blue", "expected blue in result")
	require.Contains(t, strings.ToLower(result), "orange", "expected orange in result")
	require.Contains(t, strings.ToLower(result), "purple", "expected purple in result")
	require.Contains(t, strings.ToLower(result), "yellow", "expected yellow in result")
}

func TestPineconeAsRetrieverWithMetadataFilters(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	apiKey, host := getValues(t)
	llm, e := createOpenAILLMAndEmbedder(t, rr)

	store, err := pinecone.New(
		pinecone.WithAPIKey(apiKey),
		pinecone.WithHost(host),
		pinecone.WithEmbedder(e),
	)
	require.NoError(t, err)

	id := uuid.New().String()

	_, err = store.AddDocuments(
		ctx,
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
		vectorstores.WithNameSpace(id),
	)
	require.NoError(t, err)

	waitPineconeIndexing(t)

	filter := map[string]interface{}{
		"$and": []map[string]interface{}{
			{
				"location": map[string]interface{}{
					"$in": []string{"office", "sitting room"},
				},
			},
			{
				"square_feet": map[string]interface{}{
					"$gte": 300,
				},
			},
		},
	}

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5, vectorstores.WithNameSpace(
				id), vectorstores.WithFilters(filter)),
		),
		"What color is the lamp in each room?",
	)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(result), "purple", "expected black in purple")
}
