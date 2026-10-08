package bedrockclient

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestConverseLeavesOutEmptyTextTheAPIRejects(t *testing.T) {
	t.Parallel()

	client := NewConverseClient(&MockBedrockRuntimeClient{})
	built, _, err := client.buildConverseInput(&ConverseInput{
		ModelID: "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		Messages: []Message{
			{Role: llms.ChatMessageTypeSystem, Content: "", Type: "text"},
			{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"},
		},
		Tools: []llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
			Name: "lookup", Parameters: map[string]any{"type": "object"},
		}}},
	})
	require.NoError(t, err)
	assert.Empty(t, built.System, "an empty system prompt is not a text block")
	require.Len(t, built.ToolConfig.Tools, 1)
	spec, ok := built.ToolConfig.Tools[0].(*types.ToolMemberToolSpec)
	require.True(t, ok)
	assert.Nil(t, spec.Value.Description, "an empty description stays off the tool spec")
}
