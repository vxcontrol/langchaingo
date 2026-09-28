package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func probeSend(t *testing.T, opts []Option, call ...llms.CallOption) (string, error) {
	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()
	llm, err := New(append([]Option{WithBaseURL(srv.URL), WithToken("t")}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, call...)
	return string(raw), err
}

func TestProbeSkeptic(t *testing.T) {
	for _, m := range []string{"gpt-5-codex", "gpt-5.3-codex", "my-gpt5-deployment", "gpt-5", "o3"} {
		b, err := probeSend(t, []Option{WithModel(m)}, llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithTemperature(0.2), llms.WithTopP(0.9))
		t.Logf("sampling %s err=%v body=%s", m, err, b)
	}
	for _, m := range []string{"z-ai/glm-4.6", "moonshotai/kimi-k2.5", "minimax/minimax-m2", "qwen/qwen3-235b-a22b", "x-ai/grok-code-fast-1"} {
		b, err := probeSend(t, []Option{WithModel(m), WithModernReasoningFormat()}, llms.WithReasoning(llms.ReasoningNone, 4000))
		t.Logf("budget-modern %s err=%v body=%s", m, err, b)
		b, err = probeSend(t, []Option{WithModel(m), WithModernReasoningFormat()}, llms.WithReasoning(llms.ReasoningHigh, 0))
		t.Logf("effort-modern %s err=%v body=%s", m, err, b)
		b, err = probeSend(t, []Option{WithModel(m), WithModernReasoningFormat()}, llms.WithReasoningDisabled())
		t.Logf("off-modern %s err=%v body=%s", m, err, b)
	}
	b, err := probeSend(t, []Option{WithModel("qwen/qwen3-32b")}, llms.WithReasoningDisabled())
	t.Logf("groq-off err=%v body=%s", err, b)
	b, err = probeSend(t, []Option{WithModel("qwen/qwen3-32b")}, llms.WithReasoning(llms.ReasoningHigh, 0))
	t.Logf("groq-effort err=%v body=%s", err, b)
}
