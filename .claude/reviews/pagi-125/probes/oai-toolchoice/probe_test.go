package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func probeCapture(t *testing.T, model string, extra []openai.Option, callOpts ...llms.CallOption) (map[string]any, *llms.ContentResponse, error) {
	t.Helper()
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	opts := append([]openai.Option{openai.WithBaseURL(srv.URL), openai.WithToken("x"), openai.WithModel(model)}, extra...)
	llm, err := openai.New(opts...)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, callOpts...)
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	return payload, resp, err
}

func TestProbeToolChoice(t *testing.T) {
	tools := []llms.Tool{
		{Type: "function", Function: &llms.FunctionDefinition{Name: "a", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}},
		{Type: "function", Function: &llms.FunctionDefinition{Name: "b", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}},
	}
	for name, choice := range map[string]any{
		"map-string-string-function": map[string]any{"type": "function", "function": map[string]string{"name": "b"}},
		"funcref-value":              map[string]any{"type": "function", "function": llms.FunctionReference{Name: "b"}},
		"funcref-ptr":                map[string]any{"type": "function", "function": &llms.FunctionReference{Name: "b"}},
		"map-any":                    map[string]any{"type": "function", "function": map[string]any{"name": "b"}},
		"toolchoice-struct":          llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "b"}},
	} {
		p, _, err := probeCapture(t, "gpt-4o", nil, llms.WithTools(tools), llms.WithToolChoice(choice))
		b, _ := json.Marshal(p["tool_choice"])
		t.Logf("%s: err=%v tool_choice=%s", name, err, b)
	}
}
