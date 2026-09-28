package googleai

import (
	"bytes"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

type rtFunc func(*http.Request) (*http.Response, error)

func (f rtFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestProbeNullArgsReplay(t *testing.T) {
	var calls int32
	var second string
	client := &http.Client{Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
		n := atomic.AddInt32(&calls, 1)
		body := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_time"}}]},"finishReason":"STOP"}],"usageMetadata":{}}`
		if n > 1 {
			b, _ := io.ReadAll(r.Body)
			second = string(b)
			body = `{"candidates":[{"content":{"role":"model","parts":[{"text":"noon"}]},"finishReason":"STOP"}],"usageMetadata":{}}`
		}
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(bytes.NewBufferString(body)), Request: r}, nil
	})}
	llm, err := New(t.Context(), WithAPIKey("k"), WithHTTPClient(client), WithDefaultModel("gemini-2.5-flash"))
	if err != nil {
		t.Fatal(err)
	}
	tools := []llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{Name: "get_time", Description: "time", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}}
	msgs := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time")}
	resp, err := llm.GenerateContent(t.Context(), msgs, llms.WithTools(tools))
	if err != nil {
		t.Fatal(err)
	}
	tc := resp.Choices[0].ToolCalls[0]
	t.Logf("first turn tool call arguments: %q", tc.FunctionCall.Arguments)
	msgs = append(msgs,
		llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{tc}},
		llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{ToolCallID: tc.ID, Name: "get_time", Content: "12:00"}}},
	)
	resp2, err := llm.GenerateContent(t.Context(), msgs, llms.WithTools(tools))
	t.Logf("replay: err=%v http_calls=%d", err, atomic.LoadInt32(&calls))
	if err == nil {
		t.Logf("replay ok: %q; second request has functionCall: %v", resp2.Choices[0].Content, strings.Contains(second, "functionCall"))
	}
}
