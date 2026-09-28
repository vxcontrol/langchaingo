package anthropic_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
)

func TestEveryToolResultOfOneMessageReachesTheVendor(t *testing.T) {
	t.Parallel()

	canned := &cannedMessages{responses: []string{messageWith(`{"type":"text","text":"ok"}`)}}
	_, err := canned.serve(t).GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "look both up"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.ToolCall{ID: "a", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: "{}"}},
			llms.ToolCall{ID: "b", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: "{}"}},
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "a", Name: "lookup", Content: "1"},
			llms.ToolCallResponse{ToolCallID: "b", Name: "lookup", Content: "2"},
		}},
	}, llms.WithTools([]llms.Tool{lookupTool()}))
	require.NoError(t, err)

	var payload struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(canned.requests[0], &payload))
	last := payload.Messages[len(payload.Messages)-1]
	ids := make([]any, 0, len(last.Content))
	for _, block := range last.Content {
		ids = append(ids, block["tool_use_id"])
	}
	assert.Equal(t, "user", last.Role)
	assert.Equal(t, []any{"a", "b"}, ids)
}

func TestAToolMessageWithANonResultPartIsRefusedBeforeTheRequest(t *testing.T) {
	t.Parallel()

	canned := &cannedMessages{responses: []string{messageWith(`{"type":"text","text":"ok"}`)}}
	_, err := canned.serve(t).GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "look it up"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.ToolCall{ID: "a", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: "{}"}},
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "a", Name: "lookup", Content: "1"},
			llms.TextContent{Text: "note for the model"},
		}},
	}, llms.WithTools([]llms.Tool{lookupTool()}))

	require.ErrorIs(t, err, anthropic.ErrInvalidContentType)
	assert.Empty(t, canned.requests, "the refusal comes before the request")
}
