package openai

import (
	"cmp"
	"context"
	"encoding/json"
	"io"
	"net/http"
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
		for _, asked := range []float64{1.5, 1.0001} {
			body, warnings := hostCall(t, route.baseURL, route.model, llms.WithTemperature(asked))
			assert.InDelta(t, 1, body["temperature"], 0, route)
			if assert.Contains(t, warnings, "WithTemperature", route) {
				assert.Equal(t, llms.WarningClamp, warnings["WithTemperature"].Kind)
				assert.Equal(t, strconv.FormatFloat(asked, 'g', -1, 64), warnings["WithTemperature"].Asked)
				assert.Equal(t, "1", warnings["WithTemperature"].Sent)
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
