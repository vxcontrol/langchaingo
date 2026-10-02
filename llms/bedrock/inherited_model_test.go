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
