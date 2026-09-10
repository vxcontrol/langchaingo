package anthropic_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
)

func generateForWarnings(t *testing.T, model string, callOpts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_test","type":"message","role":"assistant",` +
			`"model":"` + model + `","content":[{"type":"text","text":"ok"}],` +
			`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(
		anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL),
		anthropic.WithModel(model),
	)
	require.NoError(t, err)

	messages := []llms.MessageContent{{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.TextPart("hi")},
	}}
	resp, err := llm.GenerateContent(t.Context(), messages, callOpts...)
	require.NoError(t, err)
	return resp
}

func warningsByOption(warnings []llms.Warning) map[string]llms.Warning {
	byOption := make(map[string]llms.Warning, len(warnings))
	for _, w := range warnings {
		byOption[w.Option] = w
	}
	return byOption
}

func TestThinkingTakesTheSamplingParamsAndSaysSo(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, "claude-sonnet-4-5",
		llms.WithReasoning(llms.ReasoningMedium, 4096),
		llms.WithMaxTokens(4096),
		llms.WithTemperature(0.2),
		llms.WithTopP(0.9),
		llms.WithTopK(40),
	)

	got := warningsByOption(resp.Warnings)

	temperature, ok := got["WithTemperature"]
	require.True(t, ok, "no temperature warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningSubstitute, temperature.Kind)
	require.Equal(t, "0.2", temperature.Asked)
	require.Equal(t, "1", temperature.Sent)

	for _, option := range []string{"WithTopP", "WithTopK"} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, llms.WarningDrop, w.Kind)
		require.Empty(t, w.Sent)
	}

	require.NotContains(t, got, "WithMaxTokens",
		"the answer limit fits the budget here, so nothing should be raised")
}

func TestAnAnswerLimitRaisedForTheBudgetIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, "claude-sonnet-4-5",
		llms.WithReasoning(llms.ReasoningMedium, 4096),
		llms.WithMaxTokens(1000),
	)

	limit, ok := warningsByOption(resp.Warnings)["WithMaxTokens"]
	require.True(t, ok, "no max-tokens warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, limit.Kind)
	require.Equal(t, "1000", limit.Asked)
	require.Equal(t, "2048", limit.Sent)
}

func TestARequestThatKeepsItsSamplingCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, "claude-sonnet-4-5",
		llms.WithTemperature(0.2),
		llms.WithMaxTokens(4096),
	)
	require.Empty(t, resp.Warnings)
}
