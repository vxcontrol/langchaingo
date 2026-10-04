package openai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func TestQwenIsSentAJSONSchemaOnlyWhereDashScopeDocumentsIt(t *testing.T) {
	t.Parallel()

	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "answer", Schema: json.RawMessage(
		`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)})
	call := func(baseURL, model string) (map[string]any, error) {
		doer := &bodyDoer{content: `{"a":"x"}`}
		llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, schema)
		if doer.body == nil {
			return nil, err
		}
		var body map[string]any
		require.NoError(t, json.Unmarshal(doer.body, &body))
		return body, err
	}

	const (
		dashScope = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
		gateway   = "http://litellm.internal/v1"
	)
	for _, tc := range []struct{ baseURL, model string }{
		{dashScope, "qwen3.6-flash"}, {dashScope, "qwen-plus"}, {dashScope, "qwen3-max"}, {dashScope, "qwq-plus"},
		{dashScope, "kimi-k3"}, {dashScope, "kimi/kimi-k3"}, {dashScope, "kimi-k2-thinking"},
		{dashScope, "Moonshot-Kimi-K2-Instruct"}, {dashScope, "qwen3.8-plus"}, {dashScope, "qwen2.5-72b-instruct"},
		{gateway, "dashscope/qwen3.6-flash"},
	} {
		body, err := call(tc.baseURL, tc.model)
		var unsupported *llms.ErrStructuredOutputUnsupported
		require.True(t, errors.As(err, &unsupported), "%s: %v", tc.model, err)
		require.Nil(t, body, "%s: refused before the network", tc.model)
	}

	for _, tc := range []struct{ baseURL, model string }{
		{dashScope, "qwen3.7-flash"}, {dashScope, "qwen3.7-plus"}, {dashScope, "qwen3.7-max"},
		{dashScope, "qwen3.8-max"}, {dashScope, "qwen3.8-flash"},
		{dashScope, "qwen3.9-plus"}, {dashScope, "qwen4-max"},
		{gateway, "openrouter/qwen/qwen3.6-flash"}, {gateway, "qwen3.6-flash"},
		{"http://vllm.internal:8000/v1", "qwen3-32b"}, {"http://localhost:11434/v1", "qwen3:32b"},
		{"https://api.moonshot.ai/v1", "kimi-k3"},
	} {
		body, err := call(tc.baseURL, tc.model)
		require.NoError(t, err, tc.model)
		format, _ := body["response_format"].(map[string]any)
		assert.Equal(t, "json_schema", format["type"], tc.model)
	}
}

func TestQwQAndQVQOnModelStudioAreSentNoJSONObject(t *testing.T) {
	t.Parallel()

	const (
		dashScope = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
		gateway   = "http://litellm.internal/v1"
	)
	stream := llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil })
	call := func(baseURL, model string, opts ...llms.CallOption) (map[string]any, *llms.ContentResponse, error) {
		doer := &streamDoer{}
		llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer), WithStructuredOutputFallback())
		resp, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what is in the picture?")},
			append([]llms.CallOption{stream}, opts...)...)
		if len(doer.bodies) == 0 {
			return nil, resp, err
		}
		var body map[string]any
		require.NoError(t, json.Unmarshal(doer.bodies[0], &body))
		return body, resp, err
	}
	warningOn := func(resp *llms.ContentResponse, option string) llms.Warning {
		for _, w := range resp.Warnings {
			if w.Option == option {
				return w
			}
		}
		return llms.Warning{}
	}

	for _, tc := range []struct{ baseURL, model string }{
		{dashScope, "qwq-plus"}, {dashScope, "qvq-max"}, {gateway, "dashscope/qwq-plus"},
	} {
		body, resp, _ := call(tc.baseURL, tc.model, answerSchema())
		require.NotNil(t, body, tc.model)
		assert.NotContains(t, body, "response_format", tc.model)
		assert.Contains(t, body["messages"].([]any)[0].(map[string]any)["content"], "JSON Schema", tc.model)
		assert.Equal(t, llms.Warning{
			Kind: llms.WarningSubstitute, Option: "WithStructuredOutput", Model: tc.model,
			Asked: "answer", Sent: "a prompt instruction",
			Reason: "the vendor has no json_object response format for this model, " +
				"so the schema travels in the prompt and the answer is validated locally",
		}, warningOn(resp, "WithStructuredOutput"), tc.model)

		body, resp, err := call(tc.baseURL, tc.model, llms.WithJSONMode())
		require.NoError(t, err, tc.model)
		assert.NotContains(t, body, "response_format", tc.model)
		assert.Equal(t, llms.WarningDrop, warningOn(resp, "WithJSONMode").Kind, tc.model)
	}

	llm := newUnitLLM(t, WithBaseURL(dashScope), WithModel("qwq-plus"), WithHTTPClient(&streamDoer{}))
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, stream, answerSchema())
	var unsupported *llms.ErrStructuredOutputUnsupported
	require.ErrorAs(t, err, &unsupported)
	assert.Equal(t, "the vendor's chat completions response_format takes neither a JSON schema nor json_object for this model",
		unsupported.Reason)

	body, _, _ := call(dashScope, "qwen3.6-plus", answerSchema())
	assert.Equal(t, map[string]any{"type": "json_object"}, body["response_format"],
		"Model Studio takes json_object from a hybrid that thinks")

	body, _, _ = call("http://vllm.internal:8000/v1", "qwq-32b", answerSchema())
	format, _ := body["response_format"].(map[string]any)
	assert.Equal(t, "json_schema", format["type"], "open weights on another host keep the schema")
}
