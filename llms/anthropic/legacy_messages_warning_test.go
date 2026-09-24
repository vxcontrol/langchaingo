package anthropic_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestTheLegacyAnthropicPathReportsTheMessagesItThrowsAway(t *testing.T) {
	t.Parallel()

	resp, err := legacyCompletionsLLM(t).GenerateContent(t.Context(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "one", "two"),
		llms.TextParts(llms.ChatMessageTypeHuman, "three"),
	})
	require.NoError(t, err)

	w, ok := warningsByOption(resp.Warnings)["messages"]
	require.True(t, ok, "two of three parts went nowhere unreported: %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, w.Kind)
	require.Equal(t, "3 parts", w.Asked)
	require.Equal(t, "1 part", w.Sent)
}

func TestASingleMessagePartOnTheLegacyPathIsNotReported(t *testing.T) {
	t.Parallel()

	resp, err := legacyCompletionsLLM(t).GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "one")})
	require.NoError(t, err)

	require.NotContains(t, warningsByOption(resp.Warnings), "messages",
		"the whole prompt travelled: %v", resp.Warnings)
}
