package openai_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func TestProbeTrcSpeed(t *testing.T) {
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","model":"gpt-4.1","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	llm, err := openai.New(openai.WithToken("k"), openai.WithBaseURL(srv.URL), openai.WithModel("gpt-4.1"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithInferenceSpeed("fast"), llms.WithSeed(3), llms.WithMinLength(5))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("body=%s", body)
	t.Logf("warnings=%+v", resp.Warnings)
}
