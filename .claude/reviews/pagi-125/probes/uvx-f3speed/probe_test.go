package openai_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func TestProbeUvxSpeed(t *testing.T) {
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"gpt-4.1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	llm, _ := openai.New(openai.WithBaseURL(srv.URL), openai.WithToken("k"), openai.WithModel("gpt-4.1"))
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithInferenceSpeed("fast"), llms.WithVerbosity("low"), llms.WithMinP(0.1))
	fmt.Println("err", err, "speedOnWire", strings.Contains(body, "speed"), "verbosityOnWire", strings.Contains(body, "verbosity"))
	fmt.Println("body", body)
	if resp != nil {
		for _, w := range resp.Warnings {
			fmt.Printf("warning: %+v\n", w)
		}
	}
}
