package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestAnUnlistedClaudeOnBedrockIsSentADisableItsReleaseWouldRefuse(t *testing.T) {
	t.Parallel()

	const legacyAnswer = `{"id":"x","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	for _, door := range []struct {
		name   string
		opts   []bedrock.Option
		answer string
	}{
		{"converse", []bedrock.Option{bedrock.WithModel("us.anthropic.claude-opus-6-v1:0"), bedrock.WithConverseAPI()}, converseAnswer},
		{"legacy", []bedrock.Option{bedrock.WithModel("us.anthropic.claude-opus-6-v1:0")}, legacyAnswer},
	} {
		resp, body := bedrockWarningsSending(t, door.answer, door.opts, llms.WithReasoningDisabled())
		thinking := body["thinking"]
		if fields, ok := body["additionalModelRequestFields"].(map[string]any); ok {
			thinking = fields["thinking"]
		}
		require.Equal(t, map[string]any{"type": "disabled"}, thinking, "%s: %v", door.name, body)
		inherited := map[string]bool{}
		for _, w := range resp.Warnings {
			if w.Kind == llms.WarningInherit {
				inherited[w.Option] = true
			}
		}
		require.True(t, inherited["WithModel"], "%s: %v", door.name, resp.Warnings)
		require.True(t, inherited["WithReasoningDisabled"], "%s: %v", door.name, resp.Warnings)
	}
}

func TestAnUnlistedClaudeOnBedrockIsSentTheDisableAndEffortItsLineDocuments(t *testing.T) {
	t.Parallel()

	const legacyAnswer = `{"id":"x","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	field := func(body map[string]any, name string) any {
		if fields, ok := body["additionalModelRequestFields"].(map[string]any); ok {
			return fields[name]
		}
		return body[name]
	}
	for _, door := range []struct {
		name   string
		opts   []bedrock.Option
		answer string
	}{
		{"converse", []bedrock.Option{bedrock.WithConverseAPI()}, converseAnswer},
		{"legacy", nil, legacyAnswer},
	} {
		opts := append([]bedrock.Option{bedrock.WithModel("us.anthropic.claude-fable-6-v1:0")}, door.opts...)
		resp, body := bedrockWarningsSending(t, door.answer, opts, llms.WithReasoningDisabled())
		require.Nil(t, field(body, "thinking"), "%s: no fable release documents a disable: %v", door.name, body)
		warning := bedrockWarningsByOption(resp.Warnings)["WithReasoningDisabled"]
		require.Equal(t, llms.WarningInherit, warning.Kind, door.name)
		require.Empty(t, warning.Sent, door.name)

		opts = append([]bedrock.Option{bedrock.WithModel("us.anthropic.claude-haiku-5-v1:0")}, door.opts...)
		resp, body = bedrockWarningsSending(t, door.answer, opts, llms.WithReasoning(llms.ReasoningMinimal, 0))
		config, _ := field(body, "output_config").(map[string]any)
		require.Equal(t, "low", config["effort"], "%s: %v", door.name, body)
		var inherited llms.Warning
		for _, w := range resp.Warnings {
			if w.Kind == llms.WarningInherit && w.Option == "WithReasoning" {
				inherited = w
			}
		}
		require.Equal(t, "low", inherited.Sent, "%s: %v", door.name, resp.Warnings)
	}
}
