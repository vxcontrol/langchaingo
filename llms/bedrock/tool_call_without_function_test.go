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

func TestAHistoryToolCallWithoutAFunctionIsRefusedBeforeTheRequestOnBedrock(t *testing.T) {
	t.Parallel()

	for _, door := range bedrockDoors("anthropic.claude-sonnet-4-5-20250929-v1:0") {
		var requests atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			requests.Add(1)
			w.WriteHeader(http.StatusInternalServerError)
		}))
		llm := bedrockLLMAgainst(t, srv, door.opts...)
		_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{ID: "call_1", Type: "function"}}},
			{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
				llms.ToolCallResponse{ToolCallID: "call_1", Name: "search", Content: "found"},
			}},
		})
		srv.Close()
		require.ErrorIs(t, err, llms.ErrInvalidRequest, door.name)
		require.Zero(t, requests.Load(), "%s: no request may go out", door.name)
	}
}
