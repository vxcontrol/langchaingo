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

func TestAToolWithoutAFunctionIsDroppedAndReportedOnBedrock(t *testing.T) {
	t.Parallel()

	bare := llms.Tool{Type: "function"}
	lookup := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}
	for _, door := range bedrockDoors("anthropic.claude-sonnet-4-5-20250929-v1:0") {
		resp, body := bedrockWarningsSending(t, door.answer, door.opts,
			llms.WithTools([]llms.Tool{bare}), llms.WithToolChoice("required"))
		require.NotContains(t, body, "tools", door.name)
		require.NotContains(t, body, "toolConfig", door.name)
		byOption := bedrockWarningsByOption(resp.Warnings)
		require.Equal(t, llms.WarningDrop, byOption["WithTools"].Kind, "%s: %v", door.name, resp.Warnings)
		require.Equal(t, "1 tools", byOption["WithTools"].Asked, door.name)
		require.Equal(t, llms.WarningDrop, byOption["WithToolChoice"].Kind, "%s: %v", door.name, resp.Warnings)

		resp, body = bedrockWarningsSending(t, door.answer, door.opts,
			llms.WithTools([]llms.Tool{bare, lookup}), llms.WithToolChoice("required"))
		require.Equal(t, "1 tools", bedrockWarningsByOption(resp.Warnings)["WithTools"].Asked, door.name)
		require.Empty(t, toolChoiceWarnings(resp.Warnings), door.name)
		if door.name == "legacy" {
			require.Len(t, body["tools"], 1)
			require.Equal(t, map[string]any{"type": "any"}, body["tool_choice"])
			continue
		}
		toolConfig, _ := body["toolConfig"].(map[string]any)
		require.Len(t, toolConfig["tools"], 1)
		require.Equal(t, map[string]any{"any": map[string]any{}}, toolConfig["toolChoice"])
	}
}

func TestAPayloadWithoutToolsReportsAToolChoiceAsConverseDoes(t *testing.T) {
	t.Parallel()

	const metaAnswer = `{"generation":"ok","stop_reason":"stop","prompt_token_count":1,` +
		`"generation_token_count":1}`
	const novaAnswer = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},` +
		`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`
	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}})
	named := llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "lookup"}}

	for _, door := range []struct {
		name, model, answer string
		converse            bool
	}{
		{"legacy meta", "meta.llama3-70b-instruct-v1:0", metaAnswer, false},
		{"legacy nova", "amazon.nova-lite-v1:0", novaAnswer, false},
		{"converse meta", "meta.llama3-70b-instruct-v1:0", converseAnswer, true},
	} {
		opts := []bedrock.Option{bedrock.WithModel(door.model)}
		if door.converse {
			opts = append(opts, bedrock.WithConverseAPI())
		}
		for _, tc := range []struct {
			choice any
			asked  string
		}{
			{"required", "any"},
			{named, "lookup"},
		} {
			resp, _ := bedrockWarningsSending(t, door.answer, opts, llms.WithToolChoice(tc.choice))
			require.Equal(t, []llms.Warning{{
				Kind: llms.WarningDrop, Option: "WithToolChoice", Model: door.model,
				Asked: tc.asked, Reason: "the request carries no tools to choose from",
			}}, toolChoiceWarnings(resp.Warnings), "%s %v", door.name, tc.choice)
		}
		for _, choice := range []any{"auto", "none"} {
			resp, _ := bedrockWarningsSending(t, door.answer, opts, llms.WithToolChoice(choice))
			require.Empty(t, toolChoiceWarnings(resp.Warnings), "%s %v", door.name, choice)
		}
		if door.converse {
			continue
		}
		resp, body := bedrockWarningsSending(t, door.answer, opts, tools, llms.WithToolChoice("required"))
		require.NotContains(t, body, "tools", door.name)
		require.Equal(t, "any", bedrockWarningsByOption(resp.Warnings)["WithToolChoice"].Asked, door.name)
	}
}
