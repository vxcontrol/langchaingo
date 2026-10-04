package anthropic_test

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
)

func TestAToolWithoutAFunctionIsDroppedAndReportedOnAnthropic(t *testing.T) {
	t.Parallel()

	named := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "search", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}
	body, resp, err := generateRecording(t, "claude-sonnet-4-5",
		llms.WithTools([]llms.Tool{{Type: "web_search_20250305"}, named}))
	require.NoError(t, err)
	tools, _ := body["tools"].([]any)
	require.Len(t, tools, 1, "%v", body)
	var dropped *llms.Warning
	for i, w := range resp.Warnings {
		if w.Option == "WithTools" {
			dropped = &resp.Warnings[i]
		}
	}
	require.NotNil(t, dropped, "%v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, dropped.Kind)
	require.Equal(t, "1 tools", dropped.Asked)
}

func TestAHistoryToolCallWithoutAFunctionIsRefusedBeforeTheRequestOnAnthropic(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"), anthropic.WithBaseURL(srv.URL),
		anthropic.WithModel("claude-sonnet-4-5"))
	require.NoError(t, err)
	_, err = llm.GenerateContent(t.Context(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{ID: "call_1", Type: "function"}}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "call_1", Name: "search", Content: "found"},
		}},
	})
	require.ErrorIs(t, err, llms.ErrInvalidRequest)
	require.Zero(t, requests.Load(), "no request may go out")
}
