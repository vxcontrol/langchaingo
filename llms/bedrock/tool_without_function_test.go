package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestBothDoorsSendClaudeOnlyTheToolsThatHaveAFunction(t *testing.T) {
	t.Parallel()

	tools := []llms.Tool{
		{Type: "function"},
		{Type: "function", Function: &llms.FunctionDefinition{Name: "lookup", Parameters: map[string]any{"type": "object"}}},
	}
	const model = "anthropic.claude-sonnet-4-5-20250929-v1:0"

	_, body := bedrockWarningsSending(t, legacyAnswer, []bedrock.Option{bedrock.WithModel(model)}, llms.WithTools(tools))
	sent, _ := body["tools"].([]any)
	require.Len(t, sent, 1, "legacy: %v", body["tools"])
	require.Equal(t, "lookup", sent[0].(map[string]any)["name"])

	_, body = bedrockWarningsSending(t, converseAnswer,
		[]bedrock.Option{bedrock.WithModel(model), bedrock.WithConverseAPI()}, llms.WithTools(tools))
	toolConfig, _ := body["toolConfig"].(map[string]any)
	sent, _ = toolConfig["tools"].([]any)
	require.Len(t, sent, 1, "converse: %v", body["toolConfig"])
	spec, _ := sent[0].(map[string]any)["toolSpec"].(map[string]any)
	require.Equal(t, "lookup", spec["name"])
}
