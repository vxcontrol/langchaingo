package openai_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func TestProbeSkQ2(t *testing.T) {
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()
	for _, m := range []string{"qwen-plus", "qwen-plus-latest", "qwen-plus-2025-07-28", "z-ai/glm-4.6"} {
		for _, o := range [][]llms.CallOption{{llms.WithReasoning(llms.ReasoningHigh, 0)}, {llms.WithReasoningDisabled()}} {
			for _, modern := range []bool{false, true} {
				oo := []openai.Option{openai.WithToken("k"), openai.WithBaseURL(srv.URL), openai.WithModel(m)}
				if modern {
					oo = append(oo, openai.WithModernReasoningFormat())
				}
				llm, err := openai.New(oo...)
				if err != nil {
					t.Fatal(err)
				}
				_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, o...)
				t.Logf("%s modern=%v err=%v body=%s", m, modern, err, body)
			}
		}
	}
}
