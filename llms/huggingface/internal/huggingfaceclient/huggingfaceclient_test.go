package huggingfaceclient

import (
	"os"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/httputil"
	"github.com/vxcontrol/langchaingo/internal/httprr"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testURL = "https://router.huggingface.co"

func TestClient_CreateEmbedding(t *testing.T) {
	t.Skip("temporary skip")
	ctx := t.Context()

	// Check both HF_TOKEN and HUGGINGFACEHUB_API_TOKEN
	if os.Getenv("HF_TOKEN") == "" && os.Getenv("HUGGINGFACEHUB_API_TOKEN") == "" {
		httprr.SkipIfNoCredentialsAndRecordingMissing(t, "HF_TOKEN")
	}

	rr := httprr.OpenForTest(t, httputil.DefaultTransport)
	defer rr.Close()

	apiKey := "test-api-key"
	if rr.Recording() {
		// Try HF_TOKEN first, then fall back to HUGGINGFACEHUB_API_TOKEN
		if key := os.Getenv("HF_TOKEN"); key != "" {
			apiKey = key
		} else if key := os.Getenv("HUGGINGFACEHUB_API_TOKEN"); key != "" {
			apiKey = key
		}
	}

	// Create client with recording HTTP client
	client, err := New(apiKey, "", testURL, WithHTTPClient(rr.Client()))
	require.NoError(t, err)

	req := &EmbeddingRequest{
		Inputs: []string{"Hello world", "How are you?"},
	}

	embeddings, err := client.CreateEmbedding(ctx, "BAAI/bge-small-en-v1.5", "feature-extraction", req)
	require.NoError(t, err)
	assert.NotNil(t, embeddings)
	assert.Len(t, embeddings, 2)
	assert.NotEmpty(t, embeddings[0])
	assert.NotEmpty(t, embeddings[1])
}

func TestClient_InvalidToken(t *testing.T) {
	_, err := New("", "model", testURL)
	assert.ErrorIs(t, err, ErrInvalidToken)
}

func TestClient_RunInferenceWithProvider(t *testing.T) {
	ctx := t.Context()

	// Check both HF_TOKEN and HUGGINGFACEHUB_API_TOKEN
	if os.Getenv("HF_TOKEN") == "" && os.Getenv("HUGGINGFACEHUB_API_TOKEN") == "" {
		httprr.SkipIfNoCredentialsAndRecordingMissing(t, "HF_TOKEN")
	}

	rr := httprr.OpenForTest(t, httputil.DefaultTransport)
	defer rr.Close()

	apiKey := "test-api-key"
	if rr.Recording() {
		// Try HF_TOKEN first, then fall back to HUGGINGFACEHUB_API_TOKEN
		if key := os.Getenv("HF_TOKEN"); key != "" {
			apiKey = key
		} else if key := os.Getenv("HUGGINGFACEHUB_API_TOKEN"); key != "" {
			apiKey = key
		}
	}

	// Create client with provider and recording HTTP client
	// Using deepseek model with hyperbolic provider as shown in user's example
	client, err := New(apiKey, "deepseek-ai/DeepSeek-R1-0528", "https://router.huggingface.co",
		WithHTTPClient(rr.Client()),
		WithProvider("hyperbolic"))
	require.NoError(t, err)

	req := &InferenceRequest{
		Model:       "deepseek-ai/DeepSeek-R1-0528",
		Prompt:      "Hello, how are you?",
		Temperature: ptr(0.5),
		MaxTokens:   ptr(50),
	}

	resp, err := client.RunInference(ctx, req)

	// Skip test if the provider is not available live (404/403 error); a replay miss fails
	if err != nil && rr.Recording() && (strings.Contains(err.Error(), "404") || strings.Contains(err.Error(), "403")) {
		t.Skip("Provider not available, skipping test")
	}

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.NotEmpty(t, resp.Text)
}
