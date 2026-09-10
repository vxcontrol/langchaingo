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
	return generateForWarningsOn(t, "glm-5", callOpts...)
}

func generateForWarningsOn(t *testing.T, model string, callOpts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"` + model + `","message":{"role":"assistant","content":"hi"},` +
			`"done":true,"done_reason":"stop"}` + "\n"))
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel(model))
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

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "2048", w.Asked)
}

func TestAPlainOllamaCallCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithTemperature(0.2))
	require.Empty(t, resp.Warnings)
}

func TestAToolChoiceOfAutoIsNotALoss(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithToolChoice("auto"))
	require.Empty(t, resp.Warnings, "auto is what the door does with no tool choice at all")
}

func TestADroppedToolChoiceIsNamedNotNumbered(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithToolChoice("none"))

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithToolChoice"]
	require.True(t, ok, "no tool-choice warning in %v", resp.Warnings)
	require.Equal(t, "none", w.Asked)
}

func TestAZeroTopKIsNotALoss(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithTopK(0))
	require.Empty(t, resp.Warnings)
}

func TestExtraBodyThisDoorCannotMergeIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithExtraBody(map[string]any{"enable_thinking": false, "chat_template_kwargs": map[string]any{}}))

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithExtraBody"]
	require.True(t, ok, "no extra-body warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "chat_template_kwargs, enable_thinking", w.Asked)
}

func TestMetadataSetAfterExtraBodyStillReportsTheLoss(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithExtraBody(map[string]any{"enable_thinking": false}),
		llms.WithMetadata(map[string]any{"user": "u1"}))

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithExtraBody"]
	require.True(t, ok, "no extra-body warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "enable_thinking", w.Asked)
}
