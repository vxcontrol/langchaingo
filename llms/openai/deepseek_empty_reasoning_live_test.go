package openai

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

// TestDeepSeekTakesBackAToolTurnWithoutReasoning replays, in thinking mode, a tool
// loop whose assistant turn holds no reasoning and a call ID DeepSeek did not issue:
// a caller rewrote it, or made the turn up. DeepSeek refuses such a turn with 400
// unless reasoning_content comes back, so the adapter sends it empty.
func TestDeepSeekTakesBackAToolTurnWithoutReasoning(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"deepseek-flash", "deepseek-v4-pro"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			llm := newTestDeepSeekClient(t, WithModel(model), WithPreserveReasoningContent())
			tools := []llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the current weather for one city.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{"city": map[string]any{"type": "string"}},
					"required":   []string{"city"},
				},
			}}}
			const callID = "call_07_Z9fjqXcYkT3bQ2wRmN8pLs4v"
			history := []llms.MessageContent{
				llms.TextParts(llms.ChatMessageTypeHuman, "What is the weather in Paris? Use the tool."),
				{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
					llms.ToolCall{ID: callID, Type: "function", FunctionCall: &llms.FunctionCall{
						Name: "get_weather", Arguments: `{"city":"Paris"}`,
					}},
				}},
				{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
					llms.ToolCallResponse{ToolCallID: callID, Name: "get_weather", Content: "Sunny, 21C"},
				}},
			}

			resp, err := llm.GenerateContent(t.Context(), history, llms.WithTools(tools), llms.WithMaxTokens(400))
			require.NoError(t, err)
			require.NotEmpty(t, resp.Choices)
			assert.Contains(t, strings.ToLower(resp.Choices[0].Content), "21")
		})
	}
}
