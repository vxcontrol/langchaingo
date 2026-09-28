package ollama

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func mho2Server(t *testing.T, lines ...string) (*httptest.Server, *[]byte) {
	t.Helper()
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, l := range lines {
			_, _ = io.WriteString(w, l+"\n")
		}
	}))
	t.Cleanup(srv.Close)
	return srv, &body
}

func TestProbeTokenBudgetWarning(t *testing.T) {
	srv, body := mho2Server(t, `{"model":"glm-5","message":{"role":"assistant","content":"hi"},"done":true,"done_reason":"stop"}`)
	llm, err := New(WithServerURL(srv.URL), WithModel("glm-5"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithMaxTokens(4096), llms.WithReasoning("", 800))
	fmt.Println("err:", err)
	var req map[string]any
	_ = json.Unmarshal(*body, &req)
	fmt.Printf("wire think: %#v\n", req["think"])
	for _, w := range resp.Warnings {
		fmt.Printf("warning: %s | kind=%s sent=%q\n", w.String(), w.Kind, w.Sent)
	}
}

func TestProbeCutStreamWithFailOnTruncation(t *testing.T) {
	frame := func(c string) string {
		b, _ := json.Marshal(map[string]any{"model": "llama3", "message": map[string]any{"role": "assistant", "content": c}, "done": false})
		return string(b)
	}
	srv, _ := mho2Server(t, frame("The answer is "), frame("forty"))
	llm, err := New(WithServerURL(srv.URL), WithModel("llama3"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithFailOnTruncation())
	fmt.Printf("cut stream, non-streaming call: err=%v content=%q stop=%q truncated=%v\n",
		err, resp.Choices[0].Content, resp.Choices[0].StopReason, resp.Choices[0].Truncated)
}
