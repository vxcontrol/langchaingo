package openai

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func vf3Capture(t *testing.T, model string, clientOpts []Option, callOpts ...llms.CallOption) (string, []string) {
	t.Helper()
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	opts := append([]Option{WithBaseURL(srv.URL), WithToken("test"), WithModel(model)}, clientOpts...)
	llm, err := New(opts...)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, callOpts...)
	if err != nil {
		return body + " ERR " + err.Error(), nil
	}
	return body, vf3Warnings(resp)
}

func TestProbeVF3(t *testing.T) {
	b, _ := vf3Capture(t, "grok-4.5", nil, llms.WithMaxTokens(100))
	fmt.Println("grok-4.5 max:", b)
	b, _ = vf3Capture(t, "anthropic/claude-opus-4-5", []Option{WithModernReasoningFormat(), WithUsingReasoningMaxTokens()}, llms.WithReasoning(llms.ReasoningNone, 100), llms.WithMaxTokens(512))
	fmt.Println("budget body:", b)
	for _, m := range []string{"gpt-4o", "gpt-4.1", "gpt-5.1", "o3-mini"} {
		b, w := vf3Capture(t, m, nil, llms.WithAdaptiveReasoning(llms.ReasoningNone))
		fmt.Printf("%s adaptive-none: body=%s warnings=%v\n", m, b, w)
		b, w = vf3Capture(t, m, nil, llms.WithReasoning(llms.ReasoningHigh, 0))
		fmt.Printf("%s effort-high: body=%s warnings=%v\n", m, b, w)
	}
}
