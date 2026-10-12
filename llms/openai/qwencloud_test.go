package openai

import (
	"context"
	"encoding/json"
	"slices"
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
	assert.NotContains(t, body, "enable_thinking", "Model Studio serves Qwen3.5 open weights' thinking without a stream")
}

func TestQwenCloudGuestsGetWhatQwenCloudDocumentsForThem(t *testing.T) {
	t.Parallel()

	budget := []llms.CallOption{llms.WithReasoning(llms.ReasoningNone, 6000), llms.WithMaxTokens(8192)}
	for _, at := range []struct{ baseURL, prefix string }{
		{qwenCloudBaseURL, ""},
		{privateGatewayBaseURL, "qwen_cloud/"},
	} {
		for _, model := range []string{"deepseek-v4.1-flash", "deepseek-v4-pro-0813", "kimi-k3", "glm-5.1", "glm-5"} {
			body, err := callWithoutModel(t, at.baseURL, at.prefix+model, llms.WithJSONMode())
			require.NoError(t, err)
			assert.Equal(t, map[string]any{"type": "json_object"}, body["response_format"],
				"%s %s: QwenCloud lists json_object for it", at.baseURL, model)
		}
		for _, model := range []string{"deepseek-v4.1-flash", "deepseek-v4-pro-0813", "deepseek-v4-pro", "deepseek-v4-flash-0731"} {
			body, err := callWithoutModel(t, at.baseURL, at.prefix+model, budget...)
			require.NoError(t, err)
			assert.InDelta(t, 5461, body["thinking_budget"], 0,
				"%s %s: QwenCloud takes a thinking budget that shares the answer limit", at.baseURL, model)
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

func TestAnExtraBodyThatSwitchesThinkingOffWinsOverTheDoorsDepth(t *testing.T) {
	t.Parallel()

	off := llms.WithExtraBody(map[string]any{"enable_thinking": false})
	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}})
	for _, at := range []struct{ baseURL, model string }{
		{modelStudioBaseURL, "qwen3.8-max"},
		{qwenCloudBaseURL, "qwen3.8-max"},
		{privateGatewayBaseURL, "qwen_cloud/qwen3.8-max"},
	} {
		for _, depth := range []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithReasoning(llms.ReasoningNone, 2048)} {
			body, resp, err := callRecording(t, at.baseURL, at.model, depth, off, tools, llms.WithToolChoice("required"))
			require.NoError(t, err, "%s: the vendor rejects a depth with enable_thinking false, the door must not send both", at.baseURL)
			assert.Equal(t, false, body["enable_thinking"], at.baseURL)
			assert.NotContains(t, body, "reasoning_effort", at.baseURL)
			assert.NotContains(t, body, "thinking_budget", at.baseURL)
			assert.Equal(t, "required", body["tool_choice"], at.baseURL)
			dropped := slices.ContainsFunc(resp.Warnings, func(w llms.Warning) bool {
				return w.Kind == llms.WarningDrop && w.Option == "WithReasoning" && w.Reason == "the extra body switches thinking off"
			})
			assert.True(t, dropped, "%s: the caller learns the depth went unsent, got %+v", at.baseURL, resp.Warnings)
		}
	}

	for _, at := range []struct{ baseURL, model string }{
		{modelStudioBaseURL, "qwen3-32b"},
		{modelStudioBaseURL, "qwen3.5-27b"},
		{qwenCloudBaseURL, "qwen3-32b"},
		{privateGatewayBaseURL, "qwen_cloud/qwen3-32b"},
	} {
		for _, depth := range []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithReasoning(llms.ReasoningNone, 2048)} {
			body, resp, err := callRecording(t, at.baseURL, at.model, depth, off)
			require.NoError(t, err, "%s %s: with thinking off the call needs no stream", at.baseURL, at.model)
			assert.Equal(t, false, body["enable_thinking"], at.baseURL)
			assert.NotContains(t, body, "reasoning_effort", at.baseURL)
			assert.NotContains(t, body, "thinking_budget", at.baseURL)
			dropped := slices.ContainsFunc(resp.Warnings, func(w llms.Warning) bool {
				return w.Kind == llms.WarningDrop && w.Option == "WithReasoning" && w.Reason == "the extra body switches thinking off"
			})
			assert.True(t, dropped, "%s %s: got %+v", at.baseURL, at.model, resp.Warnings)
		}
	}

	for _, baseURL := range []string{modelStudioBaseURL, qwenCloudBaseURL} {
		body, resp, err := callRecording(t, baseURL, "glm-6", llms.WithReasoning(llms.ReasoningLow, 0), off)
		require.NoError(t, err)
		assert.NotContains(t, body, "reasoning_effort", "%s: a name that only inherits a thinking-only release honours the off", baseURL)
		dropped := slices.ContainsFunc(resp.Warnings, func(w llms.Warning) bool {
			return w.Kind == llms.WarningDrop && w.Reason == "the extra body switches thinking off"
		})
		assert.True(t, dropped, "%s: got %+v", baseURL, resp.Warnings)
	}
}

func TestAnExtraBodyOffLeavesTheDepthOfAModelThatAlwaysThinks(t *testing.T) {
	t.Parallel()

	off := llms.WithExtraBody(map[string]any{"enable_thinking": false})
	for _, at := range []struct{ baseURL, model string }{
		{modelStudioBaseURL, "glm-5.3"},
		{modelStudioBaseURL, "ZHIPU/GLM-5.3"},
		{qwenCloudBaseURL, "glm-5.3"},
		{privateGatewayBaseURL, "qwen_cloud/glm-5.3"},
		{modelStudioBaseURL, "kimi-k2.7-code"},
	} {
		for _, depth := range []llms.CallOption{llms.WithReasoning(llms.ReasoningLow, 0), llms.WithReasoning(llms.ReasoningNone, 2048)} {
			asked, _, err := callRecording(t, at.baseURL, at.model, depth)
			require.NoError(t, err)
			body, resp, err := callRecording(t, at.baseURL, at.model, depth, off)
			require.NoError(t, err)
			assert.Equal(t, false, body["enable_thinking"], "%s %s: the extra body still goes out", at.baseURL, at.model)
			delete(asked, "enable_thinking")
			delete(body, "enable_thinking")
			assert.Equal(t, asked, body, "%s %s: an off the model cannot honour leaves the depth on the wire", at.baseURL, at.model)
			for _, w := range resp.Warnings {
				assert.NotEqual(t, "the extra body switches thinking off", w.Reason, "%s %s", at.baseURL, at.model)
			}
		}
	}
}
