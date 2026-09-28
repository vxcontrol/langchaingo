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

func TestProbeVrfxOllamaBudget(t *testing.T) {
	for _, model := range []string{"glm-5", "qwen3", "gpt-oss:20b"} {
		var body []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = io.WriteString(w, `{"model":"m","message":{"role":"assistant","content":"hi"},"done":true,"done_reason":"stop"}`+"\n")
		}))
		llm, err := New(WithServerURL(srv.URL), WithModel(model))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := llm.GenerateContent(t.Context(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithMaxTokens(4096), llms.WithReasoning("", 800))
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		fmt.Printf("model=%s err=%v wire think=%#v\n", model, err, req["think"])
		if resp != nil {
			for _, w := range resp.Warnings {
				fmt.Printf("  warning: %s | kind=%s sent=%q\n", w.String(), w.Kind, w.Sent)
			}
		}
		srv.Close()
	}
}
