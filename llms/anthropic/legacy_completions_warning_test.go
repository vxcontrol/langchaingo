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
	)
	require.NoError(t, err)

	got := warningsByOption(resp.Warnings)
	for _, option := range []string{
		"WithTopK", "WithSeed", "WithN", "WithJSONMode", "WithReasoning", "WithTools",
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
