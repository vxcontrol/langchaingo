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

func legacyCompletionsLLM(t *testing.T) *anthropic.LLM {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"completion":"ok","stop_reason":"stop_sequence"}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(
		anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL),
		anthropic.WithModel("claude-2.1"),
		anthropic.WithLegacyTextCompletionsAPI(),
	)
	require.NoError(t, err)
	return llm
}

func TestTheLegacyAnthropicPathReportsWhatItCannotCarry(t *testing.T) {
	t.Parallel()

	resp, err := legacyCompletionsLLM(t).GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithTopK(40), llms.WithSeed(7), llms.WithN(2),
		llms.WithJSONMode(), llms.WithReasoning(llms.ReasoningHigh, 0),
		llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{Name: "f"}}}),
		llms.WithFunctions([]llms.FunctionDefinition{{Name: "now"}}),
	)
	require.NoError(t, err)

	got := warningsByOption(resp.Warnings)
	for _, option := range []string{
		"WithTopK", "WithSeed", "WithN", "WithJSONMode", "WithReasoning", "WithTools", "WithFunctions",
	} {
		require.Contains(t, got, option, "the legacy request has no field for it: %v", resp.Warnings)
	}
}

func TestTheLegacyAnthropicPathRefusesAnEffortItCannotName(t *testing.T) {
	t.Parallel()

	_, err := legacyCompletionsLLM(t).GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithReasoning("enormous", 0))

	require.Error(t, err, "an effort no door records must not reach the network")
}

func TestTheLegacyAnthropicPathReportsAToolChoiceAsOnEveryDoorThatSendsNoTools(t *testing.T) {
	t.Parallel()

	hi := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{Name: "lookup"}}})
	named := llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "lookup"}}
	for _, tc := range []struct {
		choice any
		asked  string
	}{
		{"required", "any"},
		{named, "lookup"},
	} {
		for _, call := range [][]llms.CallOption{
			{llms.WithToolChoice(tc.choice)},
			{llms.WithToolChoice(tc.choice), tools},
		} {
			resp, err := legacyCompletionsLLM(t).GenerateContent(t.Context(), hi, call...)
			require.NoError(t, err)
			require.Equal(t, []llms.Warning{{
				Kind: llms.WarningDrop, Option: "WithToolChoice", Model: "claude-2.1",
				Asked: tc.asked, Reason: "the request carries no tools to choose from",
			}}, warningsFor(resp.Warnings, "WithToolChoice"), "%v", tc.choice)
		}
	}

	for _, choice := range []any{"auto", "none"} {
		resp, err := legacyCompletionsLLM(t).GenerateContent(t.Context(), hi, llms.WithToolChoice(choice))
		require.NoError(t, err)
		require.Empty(t, resp.Warnings, "%v: nothing to choose from changes nothing", choice)
	}
}
