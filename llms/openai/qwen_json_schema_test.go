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
	"github.com/vxcontrol/langchaingo/llms/structuredoutput"
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

	stream := llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil })
	human := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}
	call := func(baseURL, model string, opts ...llms.CallOption) (wireRequest, *llms.ContentResponse) {
		srv := newFallbackServer(t, fallbackReply{pieces: []string{`{"answer":"42"}`}})
		resp, err := newFallbackLLM(t, srv.URL, model, onHost(t, srv, baseURL)...).GenerateContent(t.Context(), human,
			append([]llms.CallOption{stream}, opts...)...)
		require.NoError(t, err, "%s on %q", model, baseURL)
		return srv.request(t, 0), resp
	}

	for _, tc := range []struct{ baseURL, model string }{
		{dashScopeBaseURL, "qwq-plus"}, {dashScopeBaseURL, "qvq-max"}, {"", "dashscope/qwq-plus"},
	} {
		req, resp := call(tc.baseURL, tc.model, answerSchema())
		assert.Empty(t, req.ResponseFormat, tc.model)
		text, _ := messageText(t, req.Messages[0].Content)
		assert.Contains(t, text, structuredoutput.SchemaInstruction, tc.model)
		assert.Equal(t, llms.Warning{
			Kind: llms.WarningSubstitute, Option: "WithStructuredOutput", Model: tc.model,
			Asked: "answer", Sent: "a prompt instruction",
			Reason: "the vendor has no json_object response format for this model, " +
				"so the schema travels in the prompt and the answer is validated locally",
		}, warningFor(t, resp, "WithStructuredOutput"), tc.model)

		req, resp = call(tc.baseURL, tc.model, llms.WithJSONMode())
		assert.Empty(t, req.ResponseFormat, tc.model)
		assert.Equal(t, llms.WarningDrop, warningFor(t, resp, "WithJSONMode").Kind, tc.model)
	}

	srv := newFallbackServer(t, fallbackReply{pieces: []string{`{"answer":"42"}`}})
	llm, err := New(append([]Option{WithBaseURL(srv.URL), WithToken("test"), WithModel("qwq-plus")},
		onHost(t, srv, dashScopeBaseURL)...)...)
	require.NoError(t, err)
	_, err = llm.GenerateContent(t.Context(), human, stream, answerSchema())
	var unsupported *llms.ErrStructuredOutputUnsupported
	require.ErrorAs(t, err, &unsupported)
	assert.Equal(t, "the vendor's chat completions response_format takes neither a JSON schema nor json_object for this model",
		unsupported.Reason)
	assert.Zero(t, srv.requests())

	req, _ := call(dashScopeBaseURL, "qwen3.6-plus", answerSchema())
	assert.JSONEq(t, `{"type":"json_object"}`, string(req.ResponseFormat), "Model Studio takes json_object from a hybrid that thinks")

	req, _ = call("http://vllm.internal:8000/v1", "qwq-32b", answerSchema())
	assert.Contains(t, string(req.ResponseFormat), `"json_schema"`, "open weights on another host keep the schema")

	for _, tc := range []struct{ baseURL, model string }{
		{"http://vllm.internal:8000/v1", "qwq-32b"},
		{"http://ai-gateway.vercel.sh/v1", "dashscope/qwq-plus"},
	} {
		req, resp := call(tc.baseURL, tc.model, llms.WithJSONMode())
		assert.JSONEq(t, `{"type":"json_object"}`, string(req.ResponseFormat), "%s on %s", tc.model, tc.baseURL)
		for _, w := range resp.Warnings {
			assert.NotEqual(t, "WithJSONMode", w.Option, tc.model)
		}
	}
}

func TestGuestsOnModelStudioAreSentNoJSONSchema(t *testing.T) {
	t.Parallel()

	human := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}
	for _, tc := range []struct{ baseURL, model string }{
		{dashScopeBaseURL, "deepseek-v4-pro"}, {dashScopeBaseURL, "deepseek-v4-flash"},
		{dashScopeBaseURL, "glm-5.2"}, {dashScopeBaseURL, "glm-4.6"}, {"", "dashscope/deepseek-v4-pro"},
	} {
		srv := newFallbackServer(t, fallbackReply{pieces: []string{`{"answer":"42"}`}})
		llm, err := New(append([]Option{WithBaseURL(srv.URL), WithToken("test"), WithModel(tc.model)},
			onHost(t, srv, tc.baseURL)...)...)
		require.NoError(t, err)
		_, err = llm.GenerateContent(t.Context(), human, answerSchema())
		var unsupported *llms.ErrStructuredOutputUnsupported
		require.ErrorAs(t, err, &unsupported, tc.model)
		assert.Zero(t, srv.requests(), tc.model)

		_, err = newFallbackLLM(t, srv.URL, tc.model, onHost(t, srv, tc.baseURL)...).GenerateContent(t.Context(), human, answerSchema())
		require.NoError(t, err, tc.model)
		assert.JSONEq(t, `{"type":"json_object"}`, string(srv.request(t, 0).ResponseFormat), tc.model)
	}
}
