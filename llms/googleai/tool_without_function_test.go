package googleai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAToolWithoutAFunctionIsDroppedAndReportedOnGemini(t *testing.T) {
	t.Parallel()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(context.Background(), WithAPIKey("unit-test-key"), WithEndpoint(srv.URL),
		WithDefaultModel("gemini-2.5-flash"))
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
	declarations, _ := tools[0].(map[string]any)["functionDeclarations"].([]any)
	require.Len(t, declarations, 1, "%v", body)
	w := googleWarningsByOption(resp.Warnings)["WithTools"]
	require.Equal(t, llms.WarningDrop, w.Kind, "%v", resp.Warnings)
	require.Equal(t, "1 tools", w.Asked)
}

func TestAHistoryCallOrToolTheDoorCannotSendIsRefusedBeforeTheRequestOnGemini(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(context.Background(), WithAPIKey("unit-test-key"), WithEndpoint(srv.URL),
		WithDefaultModel("gemini-2.5-flash"))
	require.NoError(t, err)
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
			[]llms.CallOption{llms.WithTools([]llms.Tool{{Type: "google_search"}})},
		},
	} {
		_, err = llm.GenerateContent(context.Background(), call.messages, call.opts...)
		require.ErrorIs(t, err, llms.ErrInvalidRequest, name)
		require.Zero(t, requests.Load(), "%s: no request may go out", name)
	}
}

func TestCachingAHistoryToolCallWithoutAFunctionIsRefusedBeforeTheRequest(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	helper, err := NewCachingHelper(context.Background(), WithAPIKey("unit-test-key"), WithEndpoint(srv.URL))
	require.NoError(t, err)
	_, err = helper.CreateCachedContent(context.Background(), "gemini-2.5-flash", []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{ID: "call_1", Type: "function"}}},
	}, time.Hour, "cache")
	require.ErrorIs(t, err, llms.ErrInvalidRequest)
	require.Zero(t, requests.Load(), "no request may go out")
}
