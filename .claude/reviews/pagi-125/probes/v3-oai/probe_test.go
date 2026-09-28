package openai_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func TestProbeV3Penalty(t *testing.T) {
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	for _, m := range []string{"deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "gpt-4o"} {
		llm, err := openai.New(openai.WithToken("x"), openai.WithBaseURL(srv.URL+"/v1"), openai.WithModel(m))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithFrequencyPenalty(0.5), llms.WithPresencePenalty(0.3))
		fmt.Printf("model=%s err=%v\n  body=%s\n", m, err, body)
		if resp != nil && len(resp.Choices) > 0 {
			fmt.Printf("  genInfo warnings=%v\n", resp.Choices[0].GenerationInfo["warnings"])
		}
	}
}
