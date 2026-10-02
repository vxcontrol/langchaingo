package shared_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/httputil"
	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/googleai/vertex"
	"github.com/vxcontrol/langchaingo/llms/streaming"

	"cloud.google.com/go/auth/credentials"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newGoogleAIClient(t *testing.T, opts ...googleai.Option) *googleai.GoogleAI {
	t.Helper()

	// A test that brings its own HTTP client answers every request itself
	if bringsOwnHTTPClient(opts) {
		t.Parallel()
		llm, err := googleai.New(t.Context(), append(opts, googleai.WithAPIKey("test-api-key"))...)
		require.NoError(t, err)
		return llm
	}

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "GOOGLE_API_KEY")

	transport := http.DefaultTransport
	if baseURL := os.Getenv("GOOGLE_BASE_URL"); baseURL != "" {
		transport = &httputil.ApiKeyTransport{
			Transport: transport,
			APIKey:    os.Getenv("GOOGLE_API_KEY"),
			BaseURL:   baseURL,
		}
	}
	rr := httprr.OpenForTest(t, transport)
	if !rr.Recording() {
		t.Parallel()
	}

	// Avoid issue with different view of request bodies for Google AI SDK
	rr.ScrubReq(httprr.JsonCompactScrubBody)

	apiKey := "test-api-key"
	if rr.Recording() {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}

	// Configure client with httprr, pinned models and the key
	opts = append(opts,
		googleai.WithRest(),
		googleai.WithDefaultModel(testModel),
		googleai.WithDefaultEmbeddingModel(testEmbeddingModel),
		googleai.WithAPIKey(apiKey),
		googleai.WithHTTPClient(rr.Client()),
	)

	llm, err := googleai.New(t.Context(), opts...)
	require.NoError(t, err)

	return llm
}

func newVertexClient(t *testing.T, opts ...googleai.Option) *vertex.Vertex {
	t.Helper()

	cloud := []googleai.Option{
		googleai.WithCloudProject("test-project"),
		googleai.WithCloudLocation("us-central1"),
	}

	// A test that brings its own HTTP client answers every request itself
	if bringsOwnHTTPClient(opts) {
		t.Parallel()
		llm, err := vertex.New(t.Context(), append(opts, cloud...)...)
		require.NoError(t, err)
		return llm
	}

	// A recorded cassette replays without credentials, and a replay miss fails.
	// This helper cannot record one: genai sends no credentials over a
	// caller-supplied HTTP client, so recording needs an ADC-authenticated transport.
	if !hasExistingRecording(t) {
		if !hasCloudCredentials() {
			t.Skip("no Vertex AI cassette and no GCP project credentials " +
				"(GOOGLE_CLOUD_PROJECT and application default credentials); this helper " +
				"could not record one either, since genai sends no credentials over a caller-supplied HTTP client")
		}
		t.Skip("no Vertex AI cassette, and this helper cannot record one: genai sends no credentials " +
			"over a caller-supplied HTTP client, so recording needs an ADC-authenticated transport")
	}

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	if !rr.Recording() {
		t.Parallel()
	}

	// Configure client with httprr, pinned models and test credentials
	opts = append(opts, cloud...)
	opts = append(opts,
		googleai.WithDefaultModel(testModel),
		googleai.WithDefaultEmbeddingModel(testEmbeddingModel),
		googleai.WithHTTPClient(rr.Client()),
	)

	llm, err := vertex.New(t.Context(), opts...)
	require.NoError(t, err)

	return llm
}

// Models the cassettes are recorded with.
const (
	testModel          = "gemini-3.8-flash"
	testEmbeddingModel = "gemini-embedding-001"
)

// bringsOwnHTTPClient reports whether the options carry an HTTP client of the
// test's own, which needs neither a cassette nor credentials.
func bringsOwnHTTPClient(opts []googleai.Option) bool {
	resolved := googleai.DefaultOptions()
	for _, opt := range opts {
		opt(&resolved)
	}
	return resolved.HTTPClient != nil
}

// hasExistingRecording checks if a httprr recording exists for this test
func hasExistingRecording(t *testing.T) bool {
	t.Helper()
	path := filepath.Join("testdata", httprr.CleanFileName(t.Name())+".httprr")
	_, err := os.Stat(path)
	_, errGzip := os.Stat(path + ".gz")
	return err == nil || errGzip == nil
}

// hasCloudCredentials reports whether GOOGLE_CLOUD_PROJECT names a project and
// application default credentials can be found.
func hasCloudCredentials() bool {
	if os.Getenv("GOOGLE_CLOUD_PROJECT") == "" {
		return false
	}
	_, err := credentials.DetectDefault(&credentials.DetectOptions{
		Scopes: []string{"https://www.googleapis.com/auth/cloud-platform"},
	})
	return err == nil
}

// funcName obtains the name of the given function value, without a package
// prefix.
func funcName(f any) string {
	fullName := runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
	parts := strings.Split(fullName, ".")
	return parts[len(parts)-1]
}

// testConfigs is a list of all test functions in this file to run with both
// client types, and their client configurations.
type testConfig struct {
	testFunc func(*testing.T, llms.Model)
	opts     []googleai.Option
}

func getTestConfigs() []testConfig {
	return []testConfig{
		{testMultiContentText, nil},
		{testGenerateFromSinglePrompt, nil},
		{testMultiContentTextChatSequence, nil},
		{testMultiContentWithSystemMessage, nil},
		{testMultiContentImageLink, nil},
		{testMultiContentImageBinary, nil},
		{testEmbeddings, nil},
		{testCandidateCountSetting, nil},
		{testMaxTokensSetting, nil},
		{testTools, nil},
		{testToolsWithInterfaceRequired, nil},
		{
			testMultiContentText,
			[]googleai.Option{googleai.WithHarmThreshold(googleai.HarmBlockMediumAndAbove)},
		},
		{
			testMultiContentTextUsingTextParts,
			[]googleai.Option{googleai.WithHarmThreshold(googleai.HarmBlockMediumAndAbove)},
		},
		{testWithStreaming, nil},
		{testWithHTTPClient, getHTTPTestClientOptions()},
	}
}

func TestGoogleAIShared(t *testing.T) {
	t.Parallel()

	testConfigs := getTestConfigs()
	for idx := range testConfigs {
		c := testConfigs[idx]
		t.Run(fmt.Sprintf("%s-googleai", funcName(c.testFunc)), func(t *testing.T) {
			c.testFunc(t, newGoogleAIClient(t, c.opts...))
		})
	}
}

func TestVertexShared(t *testing.T) {
	t.Parallel()

	testConfigs := getTestConfigs()
	for idx := range testConfigs {
		c := testConfigs[idx]
		t.Run(fmt.Sprintf("%s-vertex", funcName(c.testFunc)), func(t *testing.T) {
			c.testFunc(t, newVertexClient(t, c.opts...))
		})
	}
}

func testMultiContentText(t *testing.T, llm llms.Model) {
	t.Helper()

	parts := []llms.ContentPart{
		llms.TextPart("I'm a pomeranian"),
		llms.TextPart("What kind of mammal am I?"),
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	resp, err := llm.GenerateContent(t.Context(), content)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	c1 := resp.Choices[0]
	assert.Regexp(t, "(?i)dog|carnivo|canid|canine", c1.Content)
	assert.Contains(t, c1.GenerationInfo, "output_tokens")
	assert.NotZero(t, c1.GenerationInfo["output_tokens"])
}

func testMultiContentTextUsingTextParts(t *testing.T, llm llms.Model) {
	t.Helper()

	content := llms.TextParts(
		llms.ChatMessageTypeHuman,
		"I'm a pomeranian",
		"What kind of mammal am I?",
	)

	resp, err := llm.GenerateContent(t.Context(), []llms.MessageContent{content})
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	c1 := resp.Choices[0]
	assert.Regexp(t, "(?i)dog|canid|canine", c1.Content)
}

func testGenerateFromSinglePrompt(t *testing.T, llm llms.Model) {
	t.Helper()

	prompt := "name all the planets in the solar system"
	resp, err := llms.GenerateFromSinglePrompt(t.Context(), llm, prompt)
	require.NoError(t, err)

	assert.Regexp(t, "(?i)jupiter", resp)
}

func testMultiContentTextChatSequence(t *testing.T, llm llms.Model) {
	t.Helper()

	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart("Name some countries")},
		},
		{
			Role:  llms.ChatMessageTypeAI,
			Parts: []llms.ContentPart{llms.TextPart("Spain and Lesotho")},
		},
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart("Which if these is larger?")},
		},
	}

	resp, err := llm.GenerateContent(t.Context(), content, llms.WithModel(testModel))
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	c1 := resp.Choices[0]
	assert.Regexp(t, "(?i)spain.*larger", c1.Content)
}

func testMultiContentWithSystemMessage(t *testing.T, llm llms.Model) {
	t.Helper()

	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeSystem,
			Parts: []llms.ContentPart{llms.TextPart("You are a Spanish teacher; answer in Spanish")},
		},
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart("Name the 5 most common fruits")},
		},
	}

	resp, err := llm.GenerateContent(t.Context(), content, llms.WithModel(testModel))
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	c1 := resp.Choices[0]
	checkMatch(t, c1.Content, "(manzana|naranja)")
}

func testMultiContentImageLink(t *testing.T, llm llms.Model) {
	t.Helper()

	// The client downloads the linked image itself and sends it inline, so the
	// link is served locally and a replay needs no network.
	image, err := os.ReadFile(filepath.Join("testdata", "parrot-icon.png"))
	require.NoError(t, err)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(image)
	}))
	t.Cleanup(srv.Close)

	parts := []llms.ContentPart{
		llms.ImageURLPart(srv.URL + "/parrot-icon.png"),
		llms.TextPart("describe this image in detail"),
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	resp, err := llm.GenerateContent(
		t.Context(),
		content,
		llms.WithModel(testModel),
	)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	c1 := resp.Choices[0]
	checkMatch(t, c1.Content, "parrot")
}

func testMultiContentImageBinary(t *testing.T, llm llms.Model) {
	t.Helper()

	b, err := os.ReadFile(filepath.Join("testdata", "parrot-icon.png"))
	if err != nil {
		t.Fatal(err)
	}

	parts := []llms.ContentPart{
		llms.BinaryPart("image/png", b),
		llms.TextPart("what does this image show? please use detail"),
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	resp, err := llm.GenerateContent(
		t.Context(),
		content,
		llms.WithModel(testModel),
	)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	c1 := resp.Choices[0]
	checkMatch(t, c1.Content, "parrot")
}

func testEmbeddings(t *testing.T, llm llms.Model) {
	t.Helper()

	texts := []string{"foo", "parrot", "foo"}
	emb := llm.(embeddings.EmbedderClient)
	res, err := emb.CreateEmbedding(t.Context(), texts)
	require.NoError(t, err)

	assert.Equal(t, len(texts), len(res))
	assert.NotEmpty(t, res[0])
	assert.NotEmpty(t, res[1])
	assert.Equal(t, res[0], res[2])
}

func testCandidateCountSetting(t *testing.T, llm llms.Model) {
	t.Helper()

	parts := []llms.ContentPart{
		llms.TextPart("Name five countries in Africa"),
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	{
		resp, err := llm.GenerateContent(t.Context(), content,
			llms.WithCandidateCount(1), llms.WithTemperature(1))
		require.NoError(t, err)

		assert.Len(t, resp.Choices, 1)
	}

	// TODO: test multiple candidates when the backend supports it
}

func testWithStreaming(t *testing.T, llm llms.Model) {
	t.Helper()

	content := llms.TextParts(
		llms.ChatMessageTypeHuman,
		"I'm a pomeranian",
		"Tell me more about my taxonomy",
	)

	var (
		sb         strings.Builder
		streamDone bool
	)
	resp, err := llm.GenerateContent(
		t.Context(),
		[]llms.MessageContent{content},
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			switch chunk.Type { //nolint:exhaustive
			case streaming.ChunkTypeText:
				sb.WriteString(chunk.Content)
			case streaming.ChunkTypeDone:
				streamDone = true
			default:
				// skip other chunks
			}
			return nil
		}))

	require.NoError(t, err)

	assert.True(t, streamDone)
	assert.NotEmpty(t, resp.Choices)
	c1 := resp.Choices[0]
	checkMatch(t, c1.Content, "(dog|canid)")
	checkMatch(t, sb.String(), "(dog|canid)")
}

func testTools(t *testing.T, llm llms.Model) {
	t.Helper()

	availableTools := []llms.Tool{
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "getCurrentWeather",
				Description: "Get the current weather in a given location",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city and state, e.g. San Francisco, CA",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	content := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "What is the weather like in Chicago?"),
	}
	resp, err := llm.GenerateContent(
		t.Context(),
		content,
		llms.WithTools(availableTools))
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Choices)

	c1 := resp.Choices[0]

	// Update chat history with assistant's response, with its tool calls.
	assistantResp := llms.MessageContent{
		Role: llms.ChatMessageTypeAI,
	}
	for _, tc := range c1.ToolCalls {
		assistantResp.Parts = append(assistantResp.Parts, tc)
	}
	content = append(content, assistantResp)

	// "Execute" tool calls by calling requested function
	for _, tc := range c1.ToolCalls {
		switch tc.FunctionCall.Name {
		case "getCurrentWeather":
			var args struct {
				Location string `json:"location"`
			}
			if err := json.Unmarshal([]byte(tc.FunctionCall.Arguments), &args); err != nil {
				t.Fatal(err)
			}
			if strings.Contains(args.Location, "Chicago") {
				toolResponse := llms.MessageContent{
					Role: llms.ChatMessageTypeTool,
					Parts: []llms.ContentPart{
						llms.ToolCallResponse{
							Name:    tc.FunctionCall.Name,
							Content: "64 and sunny",
						},
					},
				}
				content = append(content, toolResponse)
			}
		default:
			t.Errorf("got unexpected function call: %v", tc.FunctionCall.Name)
		}
	}

	resp, err = llm.GenerateContent(t.Context(), content, llms.WithTools(availableTools))
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Choices)

	c1 = resp.Choices[0]
	checkMatch(t, c1.Content, "(64(°F)? and sunny|64 degrees)")
	assert.Contains(t, resp.Choices[0].GenerationInfo, "input_tokens")
	assert.Contains(t, resp.Choices[0].GenerationInfo, "output_tokens")
	assert.Contains(t, resp.Choices[0].GenerationInfo, "total_tokens")
	assert.NotZero(t, resp.Choices[0].GenerationInfo["total_tokens"])
}

func testToolsWithInterfaceRequired(t *testing.T, llm llms.Model) {
	t.Helper()

	availableTools := []llms.Tool{
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "getCurrentWeather",
				Description: "Get the current weather in a given location",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city and state, e.g. San Francisco, CA",
						},
					},
					// json.Unmarshal() may return []interface{} instead of []string
					"required": []interface{}{"location"},
				},
			},
		},
	}

	content := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "What is the weather like in Chicago?"),
	}
	resp, err := llm.GenerateContent(
		t.Context(),
		content,
		llms.WithTools(availableTools))
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Choices)

	c1 := resp.Choices[0]
	assert.Contains(t, c1.GenerationInfo, "output_tokens")
	assert.NotZero(t, c1.GenerationInfo["output_tokens"])

	// Update chat history with assistant's response, with its tool calls.
	assistantResp := llms.MessageContent{
		Role: llms.ChatMessageTypeAI,
	}
	for _, tc := range c1.ToolCalls {
		assistantResp.Parts = append(assistantResp.Parts, tc)
	}
	content = append(content, assistantResp)

	// "Execute" tool calls by calling requested function
	for _, tc := range c1.ToolCalls {
		switch tc.FunctionCall.Name {
		case "getCurrentWeather":
			var args struct {
				Location string `json:"location"`
			}
			if err := json.Unmarshal([]byte(tc.FunctionCall.Arguments), &args); err != nil {
				t.Fatal(err)
			}
			if strings.Contains(args.Location, "Chicago") {
				toolResponse := llms.MessageContent{
					Role: llms.ChatMessageTypeTool,
					Parts: []llms.ContentPart{
						llms.ToolCallResponse{
							Name:    tc.FunctionCall.Name,
							Content: "64 and sunny",
						},
					},
				}
				content = append(content, toolResponse)
			}
		default:
			t.Errorf("got unexpected function call: %v", tc.FunctionCall.Name)
		}
	}

	resp, err = llm.GenerateContent(t.Context(), content, llms.WithTools(availableTools))
	require.NoError(t, err)
	assert.NotEmpty(t, resp.Choices)

	c1 = resp.Choices[0]
	checkMatch(t, c1.Content, "(64(°F)? and sunny|64 degrees)")
	assert.Contains(t, resp.Choices[0].GenerationInfo, "input_tokens")
	assert.Contains(t, resp.Choices[0].GenerationInfo, "output_tokens")
	assert.Contains(t, resp.Choices[0].GenerationInfo, "total_tokens")
	assert.NotZero(t, resp.Choices[0].GenerationInfo["total_tokens"])
}

func testMaxTokensSetting(t *testing.T, llm llms.Model) {
	t.Helper()

	parts := []llms.ContentPart{
		llms.TextPart("I'm a pomeranian"),
		llms.TextPart("Describe my taxonomy, health and care"),
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	// First, try this with a very low MaxTokens setting for such a query; expect
	// a stop reason that max of tokens was reached.
	{
		resp, err := llm.GenerateContent(t.Context(), content, llms.WithMaxTokens(24))
		require.NoError(t, err)

		require.NotEmpty(t, resp.Choices)
		c1 := resp.Choices[0]
		assert.Equal(t, "MAX_TOKENS", c1.StopReason)
		assert.True(t, c1.Truncated)
	}

	// Now, try it again with a much larger MaxTokens setting and expect to
	// finish successfully and generate a response. The model's thoughts count
	// against the setting too.
	{
		resp, err := llm.GenerateContent(t.Context(), content, llms.WithMaxTokens(8192))
		require.NoError(t, err)

		assert.NotEmpty(t, resp.Choices)
		c1 := resp.Choices[0]
		checkMatch(t, c1.StopReason, "stop")
		checkMatch(t, c1.Content, "(dog|breed|canid|canine)")
	}
}

func testWithHTTPClient(t *testing.T, llm llms.Model) {
	t.Helper()

	resp, err := llm.GenerateContent(
		t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "testing")},
	)
	require.NoError(t, err)
	require.EqualValues(t, "test-ok", resp.Choices[0].Content)
}

func getHTTPTestClientOptions() []googleai.Option {
	client := &http.Client{Transport: &testRequestInterceptor{}}
	return []googleai.Option{googleai.WithRest(), googleai.WithHTTPClient(client)}
}

type testRequestInterceptor struct{}

func (i *testRequestInterceptor) RoundTrip(req *http.Request) (*http.Response, error) {
	defer req.Body.Close()
	content := `{
	"candidates": [{
		"content": {
			"parts": [{"text": "test-ok"}]
		},
		"finishReason": "STOP"
	}],
	"usageMetadata": {
		"promptTokenCount": 7,
		"candidatesTokenCount": 7,
		"totalTokenCount": 14
	}
}`

	resp := &http.Response{
		StatusCode: http.StatusOK, Request: req,
		Body:   io.NopCloser(bytes.NewBufferString(content)),
		Header: http.Header{},
	}
	resp.Header.Set("Content-Type", "application/json")
	return resp, nil
}

// checkMatch is a testing helper that checks `got` for regexp matches vs.
// `wants`. Each of `wants` has to match.
func checkMatch(t *testing.T, got string, wants ...string) {
	t.Helper()

	for _, want := range wants {
		re, err := regexp.Compile("(?i:" + want + ")")
		if err != nil {
			t.Fatal(err)
		}
		if !re.MatchString(got) {
			t.Errorf("\ngot %q\nwanted to match %q", got, want)
		}
	}
}
