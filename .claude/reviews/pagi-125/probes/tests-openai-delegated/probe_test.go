package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func probeSend(t *testing.T, model string, opts ...llms.CallOption) (*llms.ContentResponse, string, error) {
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	llm, err := New(WithBaseURL(srv.URL), WithToken("t"), WithModel(model))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	return resp, body, err
}

func TestProbeDelegated(t *testing.T) {
	for _, model := range []string{"gpt-4o", "gpt-4.1", "o3-mini", "gpt-5.1"} {
		for name, opt := range map[string]llms.CallOption{
			"adaptive-none": llms.WithAdaptiveReasoning(llms.ReasoningNone),
			"effort-high":   llms.WithReasoning(llms.ReasoningHigh, 0),
		} {
			resp, body, err := probeSend(t, model, opt)
			var ws []llms.Warning
			if resp != nil {
				ws = resp.Warnings
			}
			t.Logf("%s %s: err=%v warnings=%+v body=%s", model, name, err, ws, body)
		}
	}
}
