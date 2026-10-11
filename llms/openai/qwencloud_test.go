package openai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const (
	qwenCloudBaseURL      = "https://maas.qwencloudapi.com/compatible-mode/v1"
	modelStudioBaseURL    = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	privateGatewayBaseURL = "http://litellm.internal/v1"
)

func callWithoutModel(t *testing.T, baseURL, model string, opts ...llms.CallOption) (map[string]any, error) {
	t.Helper()

	body, _, err := callRecording(t, baseURL, model, opts...)
	delete(body, "model")
	return body, err
}

func TestQwenCloudIsSentWhatModelStudioIsSent(t *testing.T) {
	t.Parallel()

	human := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}})
	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "answer", Schema: json.RawMessage(
		`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)})
	for name, tc := range map[string]struct {
		model string
		opts  []llms.CallOption
		check func(t *testing.T, sent sentCall)
	}{
		"a DeepSeek guest switched off": {"deepseek-v4.1-flash", []llms.CallOption{llms.WithReasoningDisabled()},
			func(t *testing.T, sent sentCall) {
				assert.Equal(t, false, sent.body["enable_thinking"])
				assert.NotContains(t, sent.body, "thinking")
			}},
		"a DeepSeek guest with top_k": {"deepseek-v4.1-flash", []llms.CallOption{llms.WithTopK(40)},
			func(t *testing.T, sent sentCall) {
				assert.NotContains(t, sent.body, "top_k")
			}},
		"Qwen switched on": {"qwen-plus", []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)},
			func(t *testing.T, sent sentCall) {
				assert.Equal(t, true, sent.body["enable_thinking"])
			}},
		"a forced tool on a Qwen that thinks only when asked": {"qwen-plus",
			[]llms.CallOption{tools, llms.WithToolChoice("required")},
			func(t *testing.T, sent sentCall) {
				assert.Empty(t, sent.err)
				assert.Equal(t, "required", sent.body["tool_choice"])
			}},
		"a delegated depth on a Qwen that thinks only when asked": {"qwen-plus",
			[]llms.CallOption{llms.WithAdaptiveReasoning(llms.ReasoningNone)},
			func(t *testing.T, sent sentCall) {
				assert.Empty(t, sent.err)
			}},
		"a thinking budget on Qwen 3.8": {"qwen3.8-max",
			[]llms.CallOption{llms.WithReasoning(llms.ReasoningNone, 1024), llms.WithMaxTokens(8192)},
			func(t *testing.T, sent sentCall) {
				assert.InDelta(t, 1024, sent.body["thinking_budget"], 0)
				assert.NotContains(t, sent.body, "reasoning_effort")
			}},
		"a schema on Qwen 3.6": {"qwen3.6-flash", []llms.CallOption{schema},
			func(t *testing.T, sent sentCall) {
				format, _ := sent.body["response_format"].(map[string]any)
				assert.NotEqual(t, "json_schema", format["type"])
			}},
		"a forced tool while Qwen 3.8 thinks": {"qwen3.8-max",
			[]llms.CallOption{tools, llms.WithToolChoice("required"), llms.WithReasoning(llms.ReasoningHigh, 0)},
			func(t *testing.T, sent sentCall) {
				assert.Nil(t, sent.body)
				assert.NotEmpty(t, sent.err)
			}},
		"JSON mode on a GLM guest": {"glm-5.3", []llms.CallOption{llms.WithJSONMode()},
			func(t *testing.T, sent sentCall) {
				assert.NotContains(t, sent.body, "response_format")
			}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			got := sentFor(t, qwenCloudBaseURL, tc.model, human, tc.opts...)
			assert.Equal(t, sentFor(t, modelStudioBaseURL, tc.model, human, tc.opts...), got, "the QwenCloud host")
			tc.check(t, got)

			got = sentFor(t, privateGatewayBaseURL, "qwen_cloud/"+tc.model, human, tc.opts...)
			assert.Equal(t, sentFor(t, privateGatewayBaseURL, "dashscope/"+tc.model, human, tc.opts...), got,
				"the qwen_cloud route")
			tc.check(t, got)
		})
	}
}

func TestQwenCloudServesThinkingWithoutAStreamWhereItsDocsSay(t *testing.T) {
	t.Parallel()

	stream := llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil })
	for _, at := range []struct{ baseURL, prefix string }{
		{qwenCloudBaseURL, ""},
		{privateGatewayBaseURL, "qwen_cloud/"},
	} {
		body, err := callWithoutModel(t, at.baseURL, at.prefix+"qwen3.5-27b")
		require.NoError(t, err)
		assert.NotContains(t, body, "enable_thinking", "%s: QwenCloud serves Qwen3.5 open weights without a stream", at.baseURL)

		_, err = callWithoutModel(t, at.baseURL, at.prefix+"qwen3.5-27b", llms.WithReasoning(llms.ReasoningHigh, 0))
		require.NoError(t, err, at.baseURL)

		body, err = callWithoutModel(t, at.baseURL, at.prefix+"qwen3-32b")
		require.NoError(t, err)
		assert.Equal(t, false, body["enable_thinking"], "%s: Qwen3 open weights think only on a stream", at.baseURL)

		for _, opts := range [][]llms.CallOption{nil, {llms.WithReasoningDisabled()}, {llms.WithReasoning(llms.ReasoningHigh, 0)}} {
			body, err = callWithoutModel(t, at.baseURL, at.prefix+"qwen3.8-2.4t-a95b", opts...)
			var streamOnly *reasoning.ErrThinkingRequiresStream
			require.ErrorAs(t, err, &streamOnly, "%s: the model only thinks, and only on a stream", at.baseURL)
			assert.Nil(t, body)
		}
		body, err = callWithoutModel(t, at.baseURL, at.prefix+"qwen3.8-2.4t-a95b", stream)
		var streamOnly *reasoning.ErrThinkingRequiresStream
		assert.NotErrorAs(t, err, &streamOnly, at.baseURL)
		assert.Equal(t, true, body["stream"], "%s: a streaming call goes out", at.baseURL)
	}

	body, err := callWithoutModel(t, modelStudioBaseURL, "qwen3.5-27b")
	require.NoError(t, err)
	assert.Equal(t, false, body["enable_thinking"], "Model Studio serves Qwen3.5 open weights' thinking only on a stream")
}

func TestQwenCloudGuestsGetWhatQwenCloudDocumentsForThem(t *testing.T) {
	t.Parallel()

	budget := []llms.CallOption{llms.WithReasoning(llms.ReasoningNone, 1024), llms.WithMaxTokens(8192)}
	for _, at := range []struct{ baseURL, prefix string }{
		{qwenCloudBaseURL, ""},
		{privateGatewayBaseURL, "qwen_cloud/"},
	} {
		for _, model := range []string{"deepseek-v4.1-flash", "deepseek-v4-pro-0813"} {
			body, err := callWithoutModel(t, at.baseURL, at.prefix+model, llms.WithJSONMode())
			require.NoError(t, err)
			assert.Equal(t, map[string]any{"type": "json_object"}, body["response_format"],
				"%s %s: QwenCloud lists json_object for it", at.baseURL, model)

			body, err = callWithoutModel(t, at.baseURL, at.prefix+model, budget...)
			require.NoError(t, err)
			assert.InDelta(t, 1024, body["thinking_budget"], 0, "%s %s: QwenCloud takes a thinking budget", at.baseURL, model)
		}

		body, err := callWithoutModel(t, at.baseURL, at.prefix+"deepseek-v4-pro", llms.WithJSONMode())
		require.NoError(t, err)
		assert.NotContains(t, body, "response_format", "%s: QwenCloud's own copies disagree on deepseek-v4-pro", at.baseURL)

		for _, model := range []string{"glm-5.3", "kimi-k3"} {
			body, err = callWithoutModel(t, at.baseURL, at.prefix+model, llms.WithMaxTokens(1000))
			require.NoError(t, err)
			assert.InDelta(t, 1000, body["max_tokens"], 0, "%s %s: QwenCloud documents max_tokens for it", at.baseURL, model)
			assert.NotContains(t, body, "max_completion_tokens", at.baseURL)
		}
	}

	body, err := callWithoutModel(t, modelStudioBaseURL, "deepseek-v4.1-flash", llms.WithJSONMode())
	require.NoError(t, err)
	assert.NotContains(t, body, "response_format", "Model Studio lists json_object only for deepseek-v4-pro")
	body, err = callWithoutModel(t, modelStudioBaseURL, "deepseek-v4-pro", llms.WithJSONMode())
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"type": "json_object"}, body["response_format"])
	body, err = callWithoutModel(t, modelStudioBaseURL, "glm-5.3", llms.WithMaxTokens(1000))
	require.NoError(t, err)
	assert.InDelta(t, 1000, body["max_completion_tokens"], 0, "Model Studio lists max_completion_tokens for GLM")
}
