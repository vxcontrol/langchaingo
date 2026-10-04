package openai

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestFunctionsGoOutAsToolsAndAreNotReported(t *testing.T) {
	t.Parallel()

	body, warnings := hostCall(t, "https://api.openai.com/v1", "gpt-4.1",
		llms.WithFunctions([]llms.FunctionDefinition{{Name: "now", Parameters: map[string]any{"type": "object"}}}))
	require.Len(t, body["tools"], 1)
	require.NotContains(t, warnings, "WithFunctions")
}
