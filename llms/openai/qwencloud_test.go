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

	doer := &bodyDoer{content: `{"a":"x"}`}
	llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	if doer.body == nil {
		return nil, err
	}
	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	delete(body, "model")
	return body, err
}

func TestQwenCloudIsSentWhatModelStudioIsSent(t *testing.T) {
	t.Parallel()

	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}})
	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "answer", Schema: json.RawMessage(
		`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)})
	for name, tc := range map[string]struct {
		model string
		opts  []llms.CallOption
		check func(t *testing.T, body map[string]any, err error)
	}{
		"a DeepSeek guest switched off": {"deepseek-v4.1-flash", []llms.CallOption{llms.WithReasoningDisabled()},
			func(t *testing.T, body map[string]any, err error) {
				require.NoError(t, err)
				assert.Equal(t, false, body["enable_thinking"])
				assert.NotContains(t, body, "thinking")
			}},
		"a DeepSeek guest with top_k": {"deepseek-v4.1-flash", []llms.CallOption{llms.WithTopK(40)},
			func(t *testing.T, body map[string]any, err error) {
				require.NoError(t, err)
				assert.NotContains(t, body, "top_k")
			}},
		"Qwen switched on": {"qwen-plus", []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)},
			func(t *testing.T, body map[string]any, err error) {
				require.NoError(t, err)
				assert.Equal(t, true, body["enable_thinking"])
			}},
		"a thinking budget on Qwen 3.8": {"qwen3.8-max",
			[]llms.CallOption{llms.WithReasoning(llms.ReasoningNone, 1024), llms.WithMaxTokens(8192)},
			func(t *testing.T, body map[string]any, err error) {
				require.NoError(t, err)
				assert.InDelta(t, 1024, body["thinking_budget"], 0)
				assert.NotContains(t, body, "reasoning_effort")
			}},
		"a schema on Qwen 3.6": {"qwen3.6-flash", []llms.CallOption{schema},
			func(t *testing.T, body map[string]any, _ error) {
				format, _ := body["response_format"].(map[string]any)
				assert.NotEqual(t, "json_schema", format["type"])
			}},
		"a forced tool while Qwen 3.8 thinks": {"qwen3.8-max",
			[]llms.CallOption{tools, llms.WithToolChoice("required"), llms.WithReasoning(llms.ReasoningHigh, 0)},
			func(t *testing.T, body map[string]any, err error) {
				assert.Nil(t, body)
				require.Error(t, err)
			}},
		"JSON mode on a GLM guest": {"glm-5.3", []llms.CallOption{llms.WithJSONMode()},
			func(t *testing.T, body map[string]any, err error) {
				require.NoError(t, err)
				assert.NotContains(t, body, "response_format")
			}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			want, wantErr := callWithoutModel(t, modelStudioBaseURL, tc.model, tc.opts...)
			got, gotErr := callWithoutModel(t, qwenCloudBaseURL, tc.model, tc.opts...)
			assert.Equal(t, want, got, "the QwenCloud host")
			assert.IsType(t, wantErr, gotErr, "the QwenCloud host")
			tc.check(t, got, gotErr)

			want, wantErr = callWithoutModel(t, privateGatewayBaseURL, "dashscope/"+tc.model, tc.opts...)
			got, gotErr = callWithoutModel(t, privateGatewayBaseURL, "qwen_cloud/"+tc.model, tc.opts...)
			assert.Equal(t, want, got, "the qwen_cloud route")
			assert.IsType(t, wantErr, gotErr, "the qwen_cloud route")
			tc.check(t, got, gotErr)
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
