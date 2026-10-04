package anthropic_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestFunctionsTheDoorNeverSendsLeaveAForcedChoiceWithNoToolsToChooseFrom(t *testing.T) {
	t.Parallel()

	functions := llms.WithFunctions([]llms.FunctionDefinition{{Name: "now"}})
	for _, model := range []string{"claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-4-6"} {
		resp, body := generateForModelSending(t, model, functions, llms.WithToolChoice("required"))
		require.NotContains(t, body, "tools", model)
		require.NotContains(t, body, "tool_choice", model)
		byOption := warningsByOption(resp.Warnings)
		require.Equal(t, llms.WarningDrop, byOption["WithFunctions"].Kind, "%s: %v", model, resp.Warnings)
		require.Equal(t, "1 functions", byOption["WithFunctions"].Asked, model)
		require.Equal(t, llms.WarningDrop, byOption["WithToolChoice"].Kind, "%s: %v", model, resp.Warnings)
	}
}
