package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func bedrockDoors(model string) []struct {
	name   string
	answer string
	opts   []bedrock.Option
} {
	return []struct {
		name   string
		answer string
		opts   []bedrock.Option
	}{
		{"legacy", legacyAnswer, []bedrock.Option{bedrock.WithModel(model)}},
		{"converse", converseAnswer, []bedrock.Option{bedrock.WithModel(model), bedrock.WithConverseAPI()}},
	}
}

func toolChoiceWarnings(warnings []llms.Warning) []llms.Warning {
	var found []llms.Warning
	for _, w := range warnings {
		if w.Option == "WithToolChoice" {
			found = append(found, w)
		}
	}
	return found
}

func TestAForcedToolChoiceWithoutToolsIsReportedAsDroppedOnBedrock(t *testing.T) {
	t.Parallel()

	named := llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "lookup"}}
	for _, door := range bedrockDoors("anthropic.claude-sonnet-4-5-20250929-v1:0") {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			for _, tc := range []struct {
				choice any
				asked  string
			}{
				{"required", "any"},
				{named, "lookup"},
			} {
				resp, body := bedrockWarningsSending(t, door.answer, door.opts, llms.WithToolChoice(tc.choice))
				require.NotContains(t, body, "tool_choice", "%v", tc.choice)
				require.NotContains(t, body, "toolConfig", "%v", tc.choice)
				reported := toolChoiceWarnings(resp.Warnings)
				require.Len(t, reported, 1, "%v: %v", tc.choice, resp.Warnings)
				require.Equal(t, llms.WarningDrop, reported[0].Kind)
				require.Equal(t, tc.asked, reported[0].Asked)
			}

			for _, choice := range []any{"auto", "none"} {
				resp, _ := bedrockWarningsSending(t, door.answer, door.opts, llms.WithToolChoice(choice))
				require.Empty(t, toolChoiceWarnings(resp.Warnings), "%v: %v", choice, resp.Warnings)
			}
		})
	}
}

func TestToolsOnlyInTheExtraBodyDoNotMakeAForcedChoiceRefusedOnBedrock(t *testing.T) {
	t.Parallel()

	extraTools := llms.WithExtraBody(map[string]any{"tools": []any{
		map[string]any{"name": "lookup", "input_schema": map[string]any{"type": "object"}},
	}})
	for _, door := range bedrockDoors("us.anthropic.claude-opus-5-5") {
		resp, body := bedrockWarningsSending(t, door.answer, door.opts, extraTools, llms.WithToolChoice("required"))
		require.NotContains(t, body, "tools", door.name)
		require.NotContains(t, body, "toolConfig", door.name)
		byOption := bedrockWarningsByOption(resp.Warnings)
		require.Contains(t, byOption, "WithExtraBody", door.name)
		require.Equal(t, llms.WarningDrop, byOption["WithToolChoice"].Kind, door.name)
	}
}
