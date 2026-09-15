package bedrockclient

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

type cannedTransport struct{ body string }

func (c *cannedTransport) Do(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(c.body)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Request:    req,
	}, nil
}

func legacyClientAnswering(body string) *Client {
	return &Client{client: bedrockruntime.New(bedrockruntime.Options{
		Region:           "us-east-1",
		Credentials:      credentials.NewStaticCredentialsProvider("k", "s", ""),
		HTTPClient:       &cannedTransport{body: body},
		RetryMaxAttempts: 1,
	})}
}

func legacyCall(t *testing.T, model string, options llms.CallOptions) *llms.ContentResponse {
	t.Helper()
	return legacyCallAnswering(t, model, `{"generation":"hi","stop_reason":"stop"}`, options)
}

func legacyCallAnswering(t *testing.T, model, body string, options llms.CallOptions) *llms.ContentResponse {
	t.Helper()

	options.Model = aws.String(model)
	resp, err := legacyClientAnswering(body).
		CreateCompletion(context.Background(), model,
			[]Message{{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"}}, options)
	require.NoError(t, err)
	return resp
}

const anthropicLegacyBody = `{"id":"msg_1","type":"message","role":"assistant",` +
	`"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn",` +
	`"usage":{"input_tokens":1,"output_tokens":1}}`

func legacyWarningsByOption(warnings []llms.Warning) map[string]llms.Warning {
	byOption := make(map[string]llms.Warning, len(warnings))
	for _, w := range warnings {
		byOption[w.Option] = w
	}
	return byOption
}

func TestALegacyPayloadWithoutAFieldReportsWhatItDropped(t *testing.T) {
	t.Parallel()

	const model = "meta.llama3-70b-instruct-v1:0"
	topK := 40
	resp := legacyCall(t, model, llms.CallOptions{
		TopK:      &topK,
		StopWords: []string{"STOP"},
		Tools:     []llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{Name: "get_weather"}}},
		ToolChoice: llms.ToolChoice{
			Type: "function", Function: &llms.FunctionReference{Name: "get_weather"},
		},
	})

	got := legacyWarningsByOption(resp.Warnings)
	for option, asked := range map[string]string{
		"WithTopK": "40", "WithStopWords": "STOP",
		"WithTools": "1 tools", "WithToolChoice": "get_weather",
	} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, llms.WarningDrop, w.Kind)
		require.Equal(t, asked, w.Asked)
		require.Equal(t, model, w.Model)
	}
}

func TestACallThatAskedForNothingCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := legacyCall(t, "meta.llama3-70b-instruct-v1:0", llms.CallOptions{})
	require.Empty(t, resp.Warnings)
}

func TestAPayloadThatCarriesTopKReportsNoTopKLoss(t *testing.T) {
	t.Parallel()

	topK := 40
	cohere := legacyClientAnswering(`{"generations":[{"text":"hi","finish_reason":"COMPLETE"}]}`)
	got, err := cohere.CreateCompletion(context.Background(), "cohere.command-text-v14",
		[]Message{{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"}},
		llms.CallOptions{Model: aws.String("cohere.command-text-v14"), TopK: &topK})
	require.NoError(t, err)
	require.NotContains(t, legacyWarningsByOption(got.Warnings), "WithTopK")
}

func TestTheLegacyAnthropicPayloadReportsWhatThinkingReshaped(t *testing.T) {
	t.Parallel()

	const model = "us.anthropic.claude-sonnet-4-5-v1:0"
	temperature, topP, topK, maxTokens := 0.2, 0.9, 40, 1000
	resp := legacyCallAnswering(t, model, anthropicLegacyBody, llms.CallOptions{
		Temperature: &temperature,
		TopP:        &topP,
		TopK:        &topK,
		MaxTokens:   &maxTokens,
		Reasoning:   &llms.ReasoningConfig{Mode: llms.ReasoningOn, Tokens: 4000},
	})

	got := legacyWarningsByOption(resp.Warnings)
	for option, sent := range map[string]string{
		"WithTemperature": "1", "WithTopP": "", "WithTopK": "", "WithMaxTokens": "2048",
	} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, model, w.Model)
		require.Equal(t, sent, w.Sent, option)
	}
}

func TestALegacyAnthropicCallThatKeepsItsValuesReportsNothing(t *testing.T) {
	t.Parallel()

	temperature, maxTokens := 0.2, 1000
	resp := legacyCallAnswering(t, "us.anthropic.claude-sonnet-4-5-v1:0", anthropicLegacyBody, llms.CallOptions{
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
	})

	require.Empty(t, resp.Warnings)
}

func TestTheLegacyDoorReportsAThinkingBudgetItCut(t *testing.T) {
	t.Parallel()

	const model = "us.anthropic.claude-sonnet-4-5-v1:0"
	maxTokens := 4096
	resp := legacyCallAnswering(t, model, anthropicLegacyBody, llms.CallOptions{
		MaxTokens: &maxTokens,
		Reasoning: &llms.ReasoningConfig{Mode: llms.ReasoningOn, Tokens: 30000},
	})

	w, ok := legacyWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, w.Kind)
	require.Equal(t, "30000 tokens", w.Asked)
	require.Equal(t, "2730 tokens", w.Sent)
}

func TestTheLegacyDoorReportsAnEffortItLowered(t *testing.T) {
	t.Parallel()

	const model = "anthropic.claude-opus-4-6-v1:0"
	maxTokens := 8000
	resp := legacyCallAnswering(t, model, anthropicLegacyBody, llms.CallOptions{
		MaxTokens: &maxTokens,
		Reasoning: &llms.ReasoningConfig{Mode: llms.ReasoningOn, Effort: llms.ReasoningXHigh},
	})

	w, ok := legacyWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, w.Kind)
	require.Equal(t, "xhigh", w.Asked)
	require.Equal(t, "high", w.Sent)
}

func TestTheLegacyNovaPathReportsWhatItDrops(t *testing.T) {
	t.Parallel()

	const novaAnswer = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},` +
		`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1}}`

	const model = "us.amazon.nova-2-lite-v1:0"
	temperature, maxTokens := 0.2, 8192
	resp := legacyCallAnswering(t, model, novaAnswer, llms.CallOptions{
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
		Reasoning:   &llms.ReasoningConfig{Mode: llms.ReasoningOn, Effort: llms.ReasoningHigh, Tokens: 4096},
	})

	got := legacyWarningsByOption(resp.Warnings)
	w, ok := got["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "4096 tokens", w.Asked)
}
