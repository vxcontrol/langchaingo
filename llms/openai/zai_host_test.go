package openai

import (
	"cmp"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

type bodyDoer struct {
	body    []byte
	content string
}

func (d *bodyDoer) Do(req *http.Request) (*http.Response, error) {
	d.body, _ = io.ReadAll(req.Body)
	content, _ := json.Marshal(cmp.Or(d.content, "ok"))
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(`{"id":"x","object":"chat.completion","created":1,"model":"m",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":` + string(content) + `},` +
			`"finish_reason":"stop"}]}`)),
	}, nil
}

func clientDialing(t *testing.T, srv *httptest.Server) *http.Client {
	t.Helper()

	addr := srv.Listener.Addr().String()
	transport := &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, addr)
	}}
	t.Cleanup(transport.CloseIdleConnections)
	return &http.Client{Transport: transport}
}

func hostCall(t *testing.T, baseURL, model string, opts ...llms.CallOption) (map[string]any, map[string]llms.Warning) {
	t.Helper()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	require.NoError(t, err)
	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	warnings := map[string]llms.Warning{}
	for _, w := range resp.Warnings {
		warnings[w.Option] = w
	}
	return body, warnings
}

func TestGLMOnZAIIsSentTheTemperatureRangeZAIDocuments(t *testing.T) {
	t.Parallel()

	for _, route := range []struct{ baseURL, model string }{
		{"https://api.z.ai/api/paas/v4", "glm-4.6"},
		{"https://open.bigmodel.cn/api/paas/v4", "glm-4.6"},
		{"http://litellm.internal/v1", "zai/glm-4.6"},
	} {
		for asked, sent := range map[float64]float64{1.5: 1, 1.0001: 1, -0.5: 0} {
			body, warnings := hostCall(t, route.baseURL, route.model, llms.WithTemperature(asked))
			assert.InDelta(t, sent, body["temperature"], 0, route)
			if assert.Contains(t, warnings, "WithTemperature", route) {
				assert.Equal(t, llms.WarningClamp, warnings["WithTemperature"].Kind)
				assert.Equal(t, strconv.FormatFloat(asked, 'g', -1, 64), warnings["WithTemperature"].Asked)
				assert.Equal(t, strconv.FormatFloat(sent, 'g', -1, 64), warnings["WithTemperature"].Sent)
			}
		}

		for _, asked := range []float64{0.7, 1} {
			body, warnings := hostCall(t, route.baseURL, route.model, llms.WithTemperature(asked))
			assert.InDelta(t, asked, body["temperature"], 1e-9, route)
			assert.NotContains(t, warnings, "WithTemperature", route)
		}
	}

	for _, route := range []struct{ baseURL, model string }{
		{"http://litellm.internal/v1", "glm-4.6"},
		{"https://ai-gateway.vercel.sh/v1", "zai/glm-4.6"},
	} {
		body, warnings := hostCall(t, route.baseURL, route.model, llms.WithTemperature(1.5))
		assert.InDelta(t, 1.5, body["temperature"], 1e-9, "another host decides its own range: %v", route)
		assert.NotContains(t, warnings, "WithTemperature", route)
	}
}

func TestZAIAndMistralAreSentTheAnswerLimitTheirSchemasName(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ baseURL, model string }{
		{"https://api.z.ai/api/paas/v4", "glm-4.6"},
		{"https://open.bigmodel.cn/api/paas/v4", "glm-5.2"},
		{"https://api.mistral.ai/v1", "mistral-large-latest"},
		{"https://codestral.mistral.ai/v1", "codestral-latest"},
		{"http://litellm.internal/v1", "zai/glm-5.2"},
		{"http://litellm.internal/v1", "mistral/mistral-large-latest"},
	} {
		body, _ := hostCall(t, tc.baseURL, tc.model, llms.WithMaxTokens(1000))
		assert.InDelta(t, 1000, body["max_tokens"], 0, tc.model)
		assert.NotContains(t, body, "max_completion_tokens", tc.model)
	}

	for _, tc := range []struct{ baseURL, model string }{
		{"http://litellm.internal/v1", "glm-5.2"},
		{"https://ai-gateway.vercel.sh/v1", "zai/glm-5.2"},
	} {
		body, _ := hostCall(t, tc.baseURL, tc.model, llms.WithMaxTokens(1000))
		assert.InDelta(t, 1000, body["max_completion_tokens"], 0, tc.model)
		assert.NotContains(t, body, "max_tokens", tc.model)
	}
}

func TestZAIIsSentNoToolsWhenAskedNotToCallThem(t *testing.T) {
	t.Parallel()

	tools := llms.WithTools([]llms.Tool{astraTool()})
	for _, route := range []struct{ baseURL, model string }{
		{"https://api.z.ai/api/paas/v4", "glm-4.6"},
		{"https://open.bigmodel.cn/api/paas/v4", "glm-4.6"},
		{"http://litellm.internal/v1", "zai/glm-4.6"},
	} {
		for name, none := range map[string]llms.CallOption{
			"by name":           llms.WithToolChoice("none"),
			"as a struct":       llms.WithToolChoice(llms.ToolChoice{Type: "none"}),
			"in the extra body": llms.WithExtraBody(map[string]any{"tool_choice": "none", "do_sample": true}),
		} {
			body, warnings := hostCall(t, route.baseURL, route.model, tools, none)
			assert.NotContains(t, body, "tools", "%s %s", route.model, name)
			assert.NotContains(t, body, "tool_choice", "%s %s", route.model, name)
			assert.Equal(t, llms.Warning{
				Kind: llms.WarningSubstitute, Option: "WithToolChoice", Model: route.model, Asked: "none", Sent: "no tools",
				Reason: "Z.ai takes only tool_choice auto, so the tools stay off the request",
			}, warnings["WithToolChoice"], "%s %s", route.model, name)
		}
		body, _ := hostCall(t, route.baseURL, route.model, tools,
			llms.WithExtraBody(map[string]any{"tool_choice": "none", "do_sample": true}))
		assert.Equal(t, true, body["do_sample"], "the rest of the extra body still goes: %s", route.model)
	}

	extra := map[string]any{"tool_choice": "none"}
	hostCall(t, "https://api.z.ai/api/paas/v4", "glm-4.6", tools, llms.WithExtraBody(extra))
	assert.Equal(t, map[string]any{"tool_choice": "none"}, extra, "the caller's extra body stays as it was")

	for _, route := range []struct{ baseURL, model string }{
		{"http://vllm.internal:8000/v1", "glm-4.6"},
		{"https://ai-gateway.vercel.sh/v1", "zai/glm-4.6"},
	} {
		body, warnings := hostCall(t, route.baseURL, route.model, tools, llms.WithToolChoice("none"))
		assert.Equal(t, "none", body["tool_choice"], route.model)
		assert.Len(t, body["tools"], 1, route.model)
		assert.NotContains(t, warnings, "WithToolChoice", route.model)
	}

	body, warnings := hostCall(t, "https://api.z.ai/api/paas/v4", "glm-4.6", tools, llms.WithToolChoice("auto"))
	assert.Equal(t, "auto", body["tool_choice"])
	assert.Len(t, body["tools"], 1)
	assert.NotContains(t, warnings, "WithToolChoice")
}
