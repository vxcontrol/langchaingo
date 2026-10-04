package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestFunctionsBedrockNeverSendsLeaveAForcedChoiceWithNoToolsToChooseFrom(t *testing.T) {
	t.Parallel()

	functions := llms.WithFunctions([]llms.FunctionDefinition{{Name: "now"}})
	for _, tc := range []struct {
		model string
		opts  []llms.CallOption
	}{
		{"us.anthropic.claude-opus-5-5", nil},
		{"anthropic.claude-sonnet-4-5-20250929-v1:0", []llms.CallOption{llms.WithReasoning(llms.ReasoningNone, 2048)}},
	} {
		for _, door := range bedrockDoors(tc.model) {
			call := append([]llms.CallOption{functions, llms.WithToolChoice("required")}, tc.opts...)
			resp, body := bedrockWarningsSending(t, door.answer, door.opts, call...)
			require.NotContains(t, body, "tools", "%s %s", tc.model, door.name)
			require.NotContains(t, body, "tool_choice", "%s %s", tc.model, door.name)
			require.NotContains(t, body, "toolConfig", "%s %s", tc.model, door.name)
			byOption := bedrockWarningsByOption(resp.Warnings)
			require.Equal(t, llms.WarningDrop, byOption["WithFunctions"].Kind, "%s %s: %v", tc.model, door.name, resp.Warnings)
			require.Equal(t, "1 functions", byOption["WithFunctions"].Asked)
			require.Equal(t, llms.WarningDrop, byOption["WithToolChoice"].Kind, "%s %s: %v", tc.model, door.name, resp.Warnings)
		}
	}
}
