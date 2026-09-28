package ollama_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/ollama"
)

// Mimics ollama v0.32.5 server/routes.go ChatHandler: a model without the
// thinking capability rejects any truthy think value with 400.
func TestProbeV1Think(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		t.Logf("request body: %s", body)
		var req struct {
			Model string          `json:"model"`
			Think json.RawMessage `json:"think"`
		}
		_ = json.Unmarshal(body, &req)
		th := string(req.Think)
		if th != "" && th != "null" && th != "false" {
			w.WriteHeader(400)
			fmt.Fprintf(w, `{"error":%q}`, fmt.Sprintf("%q does not support thinking", req.Model))
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprint(w, `{"model":"llama3.2","message":{"role":"assistant","content":"hi"},"done":true,"done_reason":"stop"}`+"\n")
	}))
	defer srv.Close()
	llm, err := ollama.New(ollama.WithServerURL(srv.URL), ollama.WithModel("llama3.2"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithReasoning(llms.ReasoningHigh, 0))
	if err != nil {
		t.Logf("RESULT: error: %v", err)
		return
	}
	t.Logf("RESULT: ok content=%q", resp.Choices[0].Content)
}
