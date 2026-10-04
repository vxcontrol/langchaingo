package openai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAToolWithoutAFunctionIsDroppedAndReported(t *testing.T) {
	t.Parallel()

	named := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "search", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}
	body, warnings := hostCall(t, "http://api.openai.com/v1", "gpt-4.1",
		llms.WithTools([]llms.Tool{{Type: "function"}, named}))
	tools, _ := body["tools"].([]any)
	require.Len(t, tools, 1, "%v", body)
	require.Equal(t, llms.WarningDrop, warnings["WithTools"].Kind, "%v", warnings)
	require.Equal(t, "1 tools", warnings["WithTools"].Asked)
}

func TestAHistoryToolCallWithoutAFunctionIsRefusedBeforeTheRequest(t *testing.T) {
	t.Parallel()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL("http://api.openai.com/v1"), WithModel("gpt-4.1"), WithHTTPClient(doer))
	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{ID: "call_1", Type: "function"}}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "call_1", Name: "search", Content: "found"},
		}},
	})
	require.ErrorIs(t, err, llms.ErrInvalidRequest)
	require.Empty(t, doer.body, "no request may go out")
}
