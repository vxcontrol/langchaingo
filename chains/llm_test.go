package chains

import (
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/callbacks"
	"github.com/vxcontrol/langchaingo/httputil"
	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/prompts"

	"github.com/stretchr/testify/require"
)

type transportWithAPIKey struct {
	Key       string
	Transport http.RoundTripper
}

func (t *transportWithAPIKey) RoundTrip(req *http.Request) (*http.Response, error) {
	rt := t.Transport
	if rt == nil {
		rt = http.DefaultTransport
		if rt == nil {
			return nil, fmt.Errorf("no Transport specified or available")
		}
	}

	newReq := *req
	if t.Key != "" {
		args := newReq.URL.Query()
		args.Set("key", t.Key)
		newReq.URL.RawQuery = args.Encode()
	}

	return rt.RoundTrip(&newReq)
}

func TestLLMChain(t *testing.T) {
	ctx := t.Context()
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")

	rr := httprr.OpenForTest(t, httputil.DefaultTransport)

	// Only run tests in parallel when not recording (to avoid rate limits)
	if rr.Replaying() {
		t.Parallel()
	}

	var opts []openai.Option
	opts = append(opts, openai.WithHTTPClient(rr.Client()))

	// Use test token when replaying
	if rr.Replaying() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}

	model, err := openai.New(opts...)
	require.NoError(t, err)
	model.CallbacksHandler = callbacks.LogHandler{}

	prompt := prompts.NewPromptTemplate(
		"What is the capital of {{.country}}",
		[]string{"country"},
	)

	chain := NewLLMChain(model, prompt)

	result, err := Predict(ctx, chain,
		map[string]any{
			"country": "France",
		},
	)
	require.NoError(t, err)
	require.True(t, strings.Contains(result, "Paris"))
}

func TestLLMChainWithChatPromptTemplate(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	c := NewLLMChain(
		&testLanguageModel{},
		prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
			prompts.NewAIMessagePromptTemplate("{{.foo}}", []string{"foo"}),
			prompts.NewHumanMessagePromptTemplate("{{.boo}}", []string{"boo"}),
		}),
	)
	result, err := Predict(ctx, c, map[string]any{
		"foo": "foo",
		"boo": "boo",
	})
	require.NoError(t, err)
	require.Equal(t, "AI: foo\nHuman: boo", result)
}

func TestLLMChainWithGoogleAI(t *testing.T) {
	ctx := t.Context()
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "GOOGLE_API_KEY")

	transport := &transportWithAPIKey{
		Key:       os.Getenv("GOOGLE_API_KEY"),
		Transport: httputil.DefaultTransport,
	}
	rr := httprr.OpenForTest(t, transport)
	defer rr.Close()

	// Configure client with httprr - use test credentials when replaying
	var opts []googleai.Option
	opts = append(opts, googleai.WithRest(), googleai.WithHTTPClient(rr.Client()))

	// Avoid issue with different view of request bodies for Google AI SDK
	rr.ScrubReq(httprr.JsonCompactScrubBody)

	if rr.Replaying() {
		// Use test credentials during replay
		opts = append(opts, googleai.WithAPIKey("test-api-key"))
		// It needs to be set here because the client goes through WithHTTPClient
		transport.Key = "test-api-key"
	}

	model, err := googleai.New(ctx, opts...)
	require.NoError(t, err)
	model.CallbacksHandler = callbacks.LogHandler{}

	prompt := prompts.NewPromptTemplate(
		"What is the capital of {{.country}}",
		[]string{"country"},
	)

	chain := NewLLMChain(model, prompt)

	// chains tramples over defaults for options, so setting these options
	// explicitly is required until https://github.com/tmc/langchaingo/issues/626
	// is fully resolved.
	result, err := Predict(ctx, chain,
		map[string]any{
			"country": "France",
		},
	)
	require.NoError(t, err)
	require.True(t, strings.Contains(result, "Paris"))
}
