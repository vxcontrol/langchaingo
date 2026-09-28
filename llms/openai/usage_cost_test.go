package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func generationInfoOf(t *testing.T, usage string) map[string]any {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"gpt-4o",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":`+usage+`}`)
	}))
	t.Cleanup(srv.Close)

	llm := newUnitLLM(t, WithBaseURL(srv.URL), WithModel("gpt-4o"))

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Choices)

	return resp.Choices[0].GenerationInfo
}

func TestTheCostReachesTheCallerAsANumber(t *testing.T) {
	t.Parallel()

	info := generationInfoOf(t, `{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,`+
		`"cost_details":{"upstream_inference_prompt_cost":0.000063,"upstream_inference_completions_cost":0.00012}}`)

	require.Contains(t, info, "UpstreamInferencePromptCost")
	assert.IsType(t, float64(0), info["UpstreamInferencePromptCost"],
		"a consumer switching on float64 must see the cost")
	assert.InDelta(t, 0.000063, info["UpstreamInferencePromptCost"], 1e-9)
	assert.InDelta(t, 0.00012, info["UpstreamInferenceCompletionsCost"], 1e-9)
}

func TestACostTheVendorDidNotSendLeavesNoKeyBehind(t *testing.T) {
	t.Parallel()

	info := generationInfoOf(t, `{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}`)

	assert.NotContains(t, info, "UpstreamInferencePromptCost")
	assert.NotContains(t, info, "UpstreamInferenceCompletionsCost")
}
