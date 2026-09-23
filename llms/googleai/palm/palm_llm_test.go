package palm

import (
	"context"
	"errors"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/httputil"
	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
)

func newPalmTestLLM(t *testing.T) *LLM {
	t.Helper()

	// No recording exists or can be made for any test here: this client talks
	// gRPC to Vertex AI (palmclient dials a gRPC connection pool and drops any
	// HTTP client), and httprr records HTTP only. The text tests would fail live
	// as well, since Google has retired text-bison, the PaLM text model they
	// call; the embedding tests call text-embedding-005, which is not retired.
	if !hasExistingRecording(t) {
		t.Skip("this client talks gRPC to Vertex AI, which httprr cannot record, so no recording exists " +
			"or can be made (the text tests also call text-bison, a retired PaLM model)")
	}

	// Temporarily unset Google API key environment variable to prevent bypass
	oldKey := os.Getenv("GOOGLE_API_KEY")
	os.Unsetenv("GOOGLE_API_KEY")
	t.Cleanup(func() {
		if oldKey != "" {
			os.Setenv("GOOGLE_API_KEY", oldKey)
		}
	})

	// Use httputil.DefaultTransport - httprr handles wrapping
	rr := httprr.OpenForTest(t, httputil.DefaultTransport)

	// Scrub auth headers
	rr.ScrubReq(func(req *http.Request) error {
		if auth := req.Header.Get("Authorization"); auth != "" {
			req.Header.Set("Authorization", "Bearer test-token")
		}
		return nil
	})

	// Set test credentials
	os.Setenv("GOOGLE_CLOUD_PROJECT", "test-project")
	os.Setenv("GOOGLE_CLOUD_LOCATION", "test-location")

	llm, err := New(WithHTTPClient(rr.Client()))
	require.NoError(t, err)

	return llm
}

// hasExistingRecording checks if a httprr recording exists for this test
func hasExistingRecording(t *testing.T) bool {
	testName := strings.ReplaceAll(t.Name(), "/", "_")
	testName = strings.ReplaceAll(testName, " ", "_")
	recordingPath := filepath.Join("testdata", testName+".httprr")
	_, err := os.Stat(recordingPath)
	return err == nil
}

func TestPaLMCall(t *testing.T) {
	t.Parallel()

	llm := newPalmTestLLM(t)

	output, err := llm.Call(t.Context(), "What is the capital of France?")
	require.NoError(t, err)
	assert.NotEmpty(t, output)
	assert.Contains(t, output, "Paris")
}

func TestPaLMGenerateContent(t *testing.T) {
	t.Parallel()

	llm := newPalmTestLLM(t)

	content := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Tell me a joke about programming"),
			},
		},
	}

	resp, err := llm.GenerateContent(t.Context(), content)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.NotEmpty(t, resp.Choices)
	assert.NotEmpty(t, resp.Choices[0].Content)
}

func TestPaLMCreateEmbedding(t *testing.T) {
	t.Parallel()

	llm := newPalmTestLLM(t)

	texts := []string{"hello world", "goodbye world", "hello world"}
	embeddings, err := llm.CreateEmbedding(t.Context(), texts)
	require.NoError(t, err)
	assert.Len(t, embeddings, 3)
	assert.NotEmpty(t, embeddings[0])
	assert.NotEmpty(t, embeddings[1])
	assert.NotEmpty(t, embeddings[2])
	// First and third should be identical since they're the same text
	assert.Equal(t, embeddings[0], embeddings[2])
}

func TestPaLMWithOptions(t *testing.T) {
	t.Parallel()

	llm := newPalmTestLLM(t)

	content := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Count from 1 to 5"),
			},
		},
	}

	resp, err := llm.GenerateContent(
		t.Context(),
		content,
		llms.WithMaxTokens(100),
		llms.WithTemperature(0.2),
	)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.NotEmpty(t, resp.Choices)
}

// TestPaLMRequestTarget pins where a request goes: the regional host of the
// location, and a model path under the project and the location. The request
// stops in an interceptor, so nothing leaves the process.
func TestPaLMRequestTarget(t *testing.T) {
	t.Parallel()

	errStopped := errors.New("stopped before sending")
	var target, resource string
	intercept := func(_ context.Context, _ string, req, _ any, cc *grpc.ClientConn,
		_ grpc.UnaryInvoker, _ ...grpc.CallOption,
	) error {
		target = cc.Target()
		if predict, ok := req.(*aiplatformpb.PredictRequest); ok {
			resource = predict.GetEndpoint()
		}
		return errStopped
	}
	noDial := func(context.Context, string) (net.Conn, error) {
		return nil, errStopped
	}

	llm, err := New(
		WithProjectID("my-project"),
		WithLocation("europe-west4"),
		WithAPIKey("test-api-key"),
		WithGRPCDialOption(grpc.WithContextDialer(noDial)),
		WithGRPCDialOption(grpc.WithChainUnaryInterceptor(intercept)),
	)
	require.NoError(t, err)

	_, err = llm.CreateEmbedding(t.Context(), []string{"hello world"})
	require.ErrorIs(t, err, errStopped)
	assert.Equal(t, "europe-west4-aiplatform.googleapis.com:443", target)
	assert.Equal(t, "projects/my-project/locations/europe-west4/publishers/google/models/text-embedding-005", resource)
}

func TestPaLMErrorHandling(t *testing.T) {
	t.Parallel()

	// Test missing project ID
	// Empty values override GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION, so
	// the result does not depend on the environment.
	_, err := New(WithLocation("us-central1"), WithProjectID(""))
	assert.Error(t, err)
	assert.Equal(t, ErrMissingProjectID, err)

	// Test missing location
	_, err = New(WithProjectID("test-project"), WithLocation(""))
	assert.Error(t, err)
	assert.Equal(t, ErrMissingLocation, err)
}

func TestPaLMMultipleTexts(t *testing.T) {
	t.Parallel()

	llm := newPalmTestLLM(t)

	// Test with empty input
	_, err := llm.CreateEmbedding(t.Context(), []string{})
	assert.Error(t, err)
	assert.Equal(t, ErrEmptyResponse, err)

	// Test with multiple texts
	texts := []string{
		"The quick brown fox",
		"jumps over the lazy dog",
		"Machine learning is fascinating",
	}
	embeddings, err := llm.CreateEmbedding(t.Context(), texts)
	require.NoError(t, err)
	assert.Len(t, embeddings, 3)

	// Each embedding should be different (different texts)
	assert.NotEqual(t, embeddings[0], embeddings[1])
	assert.NotEqual(t, embeddings[1], embeddings[2])
}

func TestPaLMWithStopWords(t *testing.T) {
	t.Parallel()

	llm := newPalmTestLLM(t)

	content := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Count from 1 to 10"),
			},
		},
	}

	resp, err := llm.GenerateContent(
		t.Context(),
		content,
		llms.WithStopWords([]string{"5"}),
	)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.NotEmpty(t, resp.Choices)

	// Should stop at or before "5"
	output := resp.Choices[0].Content
	assert.NotContains(t, output, "6")
	assert.NotContains(t, output, "7")
}
