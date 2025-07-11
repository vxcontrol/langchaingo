package chroma_test

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vxcontrol/langchaingo/chains"
	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/internal/testutil/testctr"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
	"github.com/vxcontrol/langchaingo/vectorstores/chroma"

	chromatypes "github.com/amikos-tech/chroma-go/types"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/log"
	tcchroma "github.com/testcontainers/testcontainers-go/modules/chroma"
	"github.com/testcontainers/testcontainers-go/wait"
)

// TODO (noodnik2):
//  add relevant tests from "weaviate_test.go" (the initial tests are based upon those found in "pinecone_test.go")
//  consider refactoring out standard set of vectorstore unit tests to run across all implementations

//
// NOTE: View the 'getValues()' function to see which environment variables are required to run these tests.
// WARNING: When these values are not provided, the tests will not fail, but will be (silently) skipped.
//

// createOpenAIEmbedder creates an OpenAI embedder with httprr support for testing.
func createOpenAIEmbedder(t *testing.T) *embeddings.EmbedderImpl {
	t.Helper()
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")

	openaiAPIKey := os.Getenv("OPENAI_API_KEY")
	if openaiAPIKey == "" {
		t.Skipf("Must set %s to run test", "OPENAI_API_KEY")
	}

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	llm, err := openai.New(
		openai.WithToken(openaiAPIKey),
		openai.WithEmbeddingModel("text-embedding-ada-002"),
		openai.WithHTTPClient(rr.Client()),
	)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)
	return e
}

// createOpenAILLMAndEmbedder creates both LLM and embedder with httprr support for chain tests.
func createOpenAILLMAndEmbedder(t *testing.T) (*openai.LLM, *embeddings.EmbedderImpl) {
	t.Helper()
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")

	openaiAPIKey := os.Getenv("OPENAI_API_KEY")
	if openaiAPIKey == "" {
		t.Skipf("Must set %s to run test", "OPENAI_API_KEY")
	}

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	llm, err := openai.New(
		openai.WithHTTPClient(rr.Client()),
	)
	require.NoError(t, err)
	embeddingLLM, err := openai.New(
		openai.WithToken(openaiAPIKey),
		openai.WithEmbeddingModel("text-embedding-ada-002"),
		openai.WithHTTPClient(rr.Client()),
	)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(embeddingLLM)
	require.NoError(t, err)
	return llm, e
}

func TestChromaGoStoreRest(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "CHROMA_URL")
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	_ = rr // Chroma client doesn't support custom HTTP clients
	if !rr.Recording() {
		t.Parallel()
	}

	testChromaURL := getValues(t)

	e := createOpenAIEmbedder(t)

	s, err := chroma.New(
		chroma.WithChromaURL(testChromaURL),
		chroma.WithDistanceFunction(chromatypes.COSINE),
		chroma.WithNameSpace(getTestNameSpace()),
		chroma.WithEmbedder(e),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(t, s)

	_, err = s.AddDocuments(t.Context(), []schema.Document{
		{PageContent: "tokyo", Metadata: map[string]any{
			"country": "japan",
		}},
		{PageContent: "potato"},
	})
	require.NoError(t, err)

	docs, err := s.SimilaritySearch(t.Context(), "japan", 1)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.Equal(t, "tokyo", docs[0].PageContent)
	country := docs[0].Metadata["country"]
	require.NoError(t, err)
	require.Equal(t, "japan", country)
}

func TestChromaStoreRestWithScoreThreshold(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "CHROMA_URL")
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	_ = rr // Chroma client doesn't support custom HTTP clients
	if !rr.Recording() {
		t.Parallel()
	}

	testChromaURL := getValues(t)

	e := createOpenAIEmbedder(t)

	s, err := chroma.New(
		chroma.WithChromaURL(testChromaURL),
		chroma.WithDistanceFunction(chromatypes.COSINE),
		chroma.WithNameSpace(getTestNameSpace()),
		chroma.WithEmbedder(e),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(t, s)

	_, err = s.AddDocuments(t.Context(), []schema.Document{
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
	docs, err := s.SimilaritySearch(t.Context(),
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(0.8))
	require.NoError(t, err)
	require.Len(t, docs, 6)

	// test with a score threshold of 0, expected all 10 documents
	docs, err = s.SimilaritySearch(t.Context(),
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(0))
	require.NoError(t, err)
	require.Len(t, docs, 10)
}

func TestSimilaritySearchWithInvalidScoreThreshold(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "CHROMA_URL")
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	_ = rr // Chroma client doesn't support custom HTTP clients
	if !rr.Recording() {
		t.Parallel()
	}

	testChromaURL := getValues(t)

	e := createOpenAIEmbedder(t)

	s, err := chroma.New(
		chroma.WithChromaURL(testChromaURL),
		chroma.WithNameSpace(getTestNameSpace()),
		chroma.WithEmbedder(e),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(t, s)

	_, err = s.AddDocuments(t.Context(), []schema.Document{
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

	_, err = s.SimilaritySearch(t.Context(),
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(-0.8))
	require.Error(t, err)

	_, err = s.SimilaritySearch(t.Context(),
		"Which of these are cities in Japan", 10,
		vectorstores.WithScoreThreshold(1.8))
	require.Error(t, err)
}

func TestChromaAsRetriever(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "CHROMA_URL")
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	_ = rr // Chroma client doesn't support custom HTTP clients
	if !rr.Recording() {
		t.Parallel()
	}

	testChromaURL := getValues(t)

	llm, e := createOpenAILLMAndEmbedder(t)

	s, err := chroma.New(
		chroma.WithChromaURL(testChromaURL),
		chroma.WithNameSpace(getTestNameSpace()),
		chroma.WithEmbedder(e),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(t, s)

	_, err = s.AddDocuments(
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
			vectorstores.ToRetriever(s, 1),
		),
		"What color is the desk?",
	)
	require.NoError(t, err)
	require.Contains(t, strings.ToLower(result), "orange", "expected orange in result")
}

func TestChromaAsRetrieverWithScoreThreshold(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "CHROMA_URL")
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	_ = rr // Chroma client doesn't support custom HTTP clients
	if !rr.Recording() {
		t.Parallel()
	}

	testChromaURL := getValues(t)

	llm, e := createOpenAILLMAndEmbedder(t)

	s, err := chroma.New(
		chroma.WithChromaURL(testChromaURL),
		chroma.WithDistanceFunction(chromatypes.COSINE),
		chroma.WithNameSpace(getTestNameSpace()),
		chroma.WithEmbedder(e),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(t, s)

	_, err = s.AddDocuments(
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
			vectorstores.ToRetriever(s, 5, vectorstores.WithScoreThreshold(0.8)),
		),
		"What colors is each piece of furniture next to the desk?",
	)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(result), "black", "expected black in result")
	require.Contains(t, strings.ToLower(result), "beige", "expected beige in result")
}

func TestChromaAsRetrieverWithMetadataFilterEqualsClause(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "CHROMA_URL")
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	_ = rr // Chroma client doesn't support custom HTTP clients
	if !rr.Recording() {
		t.Parallel()
	}

	testChromaURL := getValues(t)

	llm, e := createOpenAILLMAndEmbedder(t)

	s, err := chroma.New(
		chroma.WithChromaURL(testChromaURL),
		chroma.WithNameSpace(getTestNameSpace()),
		chroma.WithEmbedder(e),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(t, s)

	_, err = s.AddDocuments(
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
			{
				PageContent: "The color of the lamp beside the desk is yellow.",
				Metadata: map[string]any{
					"location": "patio",
				},
			},
		},
	)
	require.NoError(t, err)

	filter := make(map[string]any)
	filterValue := make(map[string]any)
	filterValue["$eq"] = "patio"
	filter["location"] = filterValue

	result, err := chains.Run(
		t.Context(),
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(s, 5, vectorstores.WithFilters(filter)),
		),
		"What colors is the lamp?",
	)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(result), "yellow", "expected yellow in result")
}

func TestChromaAsRetrieverWithMetadataFilterInClause(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "CHROMA_URL")
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	_ = rr // Chroma client doesn't support custom HTTP clients
	if !rr.Recording() {
		t.Parallel()
	}

	testChromaURL := getValues(t)

	llm, e := createOpenAILLMAndEmbedder(t)

	s, newChromaErr := chroma.New(
		chroma.WithChromaURL(testChromaURL),
		chroma.WithEmbedder(e),
	)
	require.NoError(t, newChromaErr)

	ns := getTestNameSpace()

	defer cleanupTestArtifacts(t, s)

	_, addDocumentsErr := s.AddDocuments(
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
			{
				PageContent: "The color of the lamp beside the desk is yellow.",
				Metadata: map[string]any{
					"location": "patio",
				},
			},
		},
		vectorstores.WithNameSpace(ns),
	)
	require.NoError(t, addDocumentsErr)

	filter := make(map[string]any)
	filterValue := make(map[string]any)
	filterValue["$in"] = []string{"office", "kitchen"}
	filter["location"] = filterValue

	result, runChainErr := chains.Run(
		t.Context(),
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(s, 5, vectorstores.WithNameSpace(ns),
				vectorstores.WithFilters(filter)),
		),
		"What color(s) was/were the lamp(s) beside the desk described as?",
	)
	require.NoError(t, runChainErr)

	require.Contains(t, strings.ToLower(result), "black", "expected black in result")
	require.Contains(t, strings.ToLower(result), "orange", "expected orange in result")
}

func TestChromaAsRetrieverWithMetadataFilterNotSelected(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "CHROMA_URL")
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	_ = rr // Chroma client doesn't support custom HTTP clients
	if !rr.Recording() {
		t.Parallel()
	}

	testChromaURL := getValues(t)

	llm, e := createOpenAILLMAndEmbedder(t)

	s, err := chroma.New(
		chroma.WithChromaURL(testChromaURL),
		chroma.WithNameSpace(getTestNameSpace()),
		chroma.WithEmbedder(e),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(t, s)

	_, err = s.AddDocuments(
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
		t.Context(),
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(s, 5),
		),
		"What are all the colors of the lamps beside the desk?",
	)
	result = strings.ToLower(result)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(result), "black", "expected black in result")
	require.Contains(t, strings.ToLower(result), "blue", "expected blue in result")
	require.Contains(t, strings.ToLower(result), "orange", "expected orange in result")
	require.Contains(t, strings.ToLower(result), "purple", "expected purple in result")
	require.Contains(t, strings.ToLower(result), "yellow", "expected yellow in result")
}

func TestChromaAsRetrieverWithMetadataFilters(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "CHROMA_URL")
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	_ = rr // Chroma client doesn't support custom HTTP clients
	if !rr.Recording() {
		t.Parallel()
	}

	testChromaURL := getValues(t)

	llm, e := createOpenAILLMAndEmbedder(t)

	s, err := chroma.New(
		chroma.WithChromaURL(testChromaURL),
		chroma.WithNameSpace(getTestNameSpace()),
		chroma.WithEmbedder(e),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(t, s)

	_, err = s.AddDocuments(
		t.Context(),
		[]schema.Document{
			{
				PageContent: "The color of the lamp beside the desk is orange.",
				Metadata: map[string]any{
					"location":    "sitting room",
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

	filter := map[string]interface{}{
		"$and": []map[string]interface{}{
			{
				"location": map[string]interface{}{
					"$eq": "sitting room",
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
		t.Context(),
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(s, 5, vectorstores.WithFilters(filter)),
		),
		"What color is the lamp beside the desk?",
	)
	require.NoError(t, err)

	result = strings.ToLower(result)
	require.Contains(t, result, "purple", "expected purple in result")
}

func getValues(t *testing.T) string {
	t.Helper()
	testctr.SkipIfDockerNotAvailable(t)

	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	ctx := t.Context()
	chromaURL := os.Getenv(chroma.ChromaURLKeyEnvVarName)
	if chromaURL == "" {
		chromaContainer, err := tcchroma.Run(
			ctx,
			"chromadb/chroma:0.4.24",
			testcontainers.WithLogger(log.TestLogger(t)),
			testcontainers.WithWaitStrategy(
				wait.ForLog("Application startup complete").
					WithStartupTimeout(30*time.Second),
			),
		)
		if err != nil && strings.Contains(err.Error(), "Cannot connect to the Docker daemon") {
			t.Skip("Docker not available")
		}
		require.NoError(t, err)

		t.Cleanup(func() {
			if err := chromaContainer.Terminate(context.Background()); err != nil { //nolint:usetesting
				t.Logf("Failed to terminate chroma container: %v", err)
			}
		})

		chromaURL, err = chromaContainer.RESTEndpoint(ctx)
		if err != nil {
			t.Skipf("Failed to get chroma container REST endpoint: %s", err)
		}
	}

	return chromaURL
}

func cleanupTestArtifacts(t *testing.T, s chroma.Store) {
	t.Helper()

	require.NoError(t, s.RemoveCollection())
}

func getTestNameSpace() string {
	return fmt.Sprintf("test-namespace-%s", uuid.New().String())
}
