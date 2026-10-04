package mistral

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAToolWithoutAFunctionIsDroppedAndReportedOnMistral(t *testing.T) {
	t.Parallel()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithAPIKey("test"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)
	named := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "search", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithTools([]llms.Tool{{Type: "function"}, named}))
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

func TestAHistoryToolCallWithoutAFunctionIsRefusedBeforeTheRequestOnMistral(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithAPIKey("test"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)
	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{ID: "call_1", Type: "function"}}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "call_1", Name: "search", Content: "found"},
		}},
	})
	require.ErrorIs(t, err, llms.ErrInvalidRequest)
	require.Zero(t, requests.Load(), "no request may go out")
}
