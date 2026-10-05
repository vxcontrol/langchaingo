package bedrock_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAHistoryCallOrToolTheDoorCannotSendIsRefusedBeforeTheRequestOnBedrock(t *testing.T) {
	t.Parallel()

	for _, door := range bedrockDoors("anthropic.claude-sonnet-4-5-20250929-v1:0") {
		var requests atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			requests.Add(1)
			w.WriteHeader(http.StatusInternalServerError)
		}))
		llm := bedrockLLMAgainst(t, srv, door.opts...)
		for name, call := range map[string]struct {
			messages []llms.MessageContent
			opts     []llms.CallOption
		}{
			"a history call without a function": {[]llms.MessageContent{
				llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
				{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{ID: "call_1", Type: "function"}}},
				{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
					llms.ToolCallResponse{ToolCallID: "call_1", Name: "search", Content: "found"},
				}},
			}, nil},
			"a built-in tool the door cannot send": {
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
				[]llms.CallOption{llms.WithTools([]llms.Tool{{Type: "web_search"}})},
			},
		} {
			_, err := llm.GenerateContent(context.Background(), call.messages, call.opts...)
			require.ErrorIs(t, err, llms.ErrInvalidRequest, "%s, %s", door.name, name)
		}
		srv.Close()
		require.Zero(t, requests.Load(), "%s: no request may go out", door.name)
	}
}
