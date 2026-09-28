package ollama

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

// fakeOld mimics an Ollama <= 0.17 server: string think values are refused for
// every model whose parser is not harmony (i.e. not gpt-oss).
func TestProbeThinkLevelOnOldServer(t *testing.T) {
	for _, model := range []string{"qwen3:8b", "deepseek-r1:8b", "gpt-oss:20b"} {
		var body string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			body = string(b)
			if strings.Contains(body, `"think":"`) && !strings.HasPrefix(model, "gpt-oss") {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":"think value \"high\" is not supported for this model"}`))
				return
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = w.Write([]byte(`{"model":"m","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}` + "\n"))
		}))
		llm, err := New(WithServerURL(srv.URL), WithModel(model))
		if err != nil {
			t.Fatal(err)
		}
		_, err = llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithReasoning(llms.ReasoningHigh, 0))
		i := strings.Index(body, `"think"`)
		think := "<absent>"
		if i >= 0 {
			think = body[i:min(len(body), i+16)]
		}
		t.Logf("model=%s think=%s err=%v", model, think, err)
		srv.Close()
	}
}
