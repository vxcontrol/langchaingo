package ollama

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func generateForWarnings(t *testing.T, callOpts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"glm-5","message":{"role":"assistant","content":"hi"},` +
			`"done":true,"done_reason":"stop"}` + "\n"))
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("glm-5"))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, callOpts...)
	require.NoError(t, err)
	return resp
}

func ollamaWarningsByOption(warnings []llms.Warning) map[string]llms.Warning {
	byOption := make(map[string]llms.Warning, len(warnings))
	for _, w := range warnings {
		byOption[w.Option] = w
	}
	return byOption
}

func TestOptionsWithNoFieldOnThisDoorAreReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithMinP(0.05), llms.WithLogProbs(true), llms.WithTopLogProbs(2),
		llms.WithN(2), llms.WithCandidateCount(3),
		llms.WithToolChoice(llms.ToolChoice{
			Type: "function", Function: &llms.FunctionReference{Name: "get_weather"},
		}))

	got := ollamaWarningsByOption(resp.Warnings)
	for option, asked := range map[string]string{
		"WithMinP": "0.05", "WithLogProbs": "true", "WithTopLogProbs": "2",
		"WithN": "2", "WithCandidateCount": "3", "WithToolChoice": "get_weather",
	} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, llms.WarningDrop, w.Kind)
		require.Equal(t, asked, w.Asked)
		require.Equal(t, "glm-5", w.Model)
	}
}

func TestAnEffortOllamaHasNoLevelForIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithReasoning(llms.ReasoningXHigh, 0))

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningSubstitute, w.Kind)
	require.Equal(t, "xhigh", w.Asked)
	require.Equal(t, "true", w.Sent)
}

func TestAThinkingTokenBudgetHasNowhereToGoOnOllama(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningHigh, 2048))

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithReasoningTokens"]
	require.True(t, ok, "no reasoning-tokens warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "2048", w.Asked)
}

func TestAPlainOllamaCallCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithTemperature(0.2))
	require.Empty(t, resp.Warnings)
}
