package openai_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

type capDoer struct{ body string }

func (c *capDoer) Do(r *http.Request) (*http.Response, error) {
	b, _ := io.ReadAll(r.Body)
	c.body = string(b)
	const completion = `{"id":"x","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}},
		Body: io.NopCloser(bytes.NewBufferString(completion)), Request: r}, nil
}

func TestProbeOpenRouterEffort(t *testing.T) {
	for _, model := range []string{"qwen/qwen3-235b-a22b", "qwen/qwen3-max", "z-ai/glm-4.6", "moonshotai/kimi-k2-thinking", "minimax/minimax-m2", "x-ai/grok-code-fast-1", "deepseek/deepseek-chat-v3.1"} {
		for _, name := range []string{"high", "off"} {
			d := &capDoer{}
			llm, err := openai.New(openai.WithBaseURL("https://openrouter.ai/api/v1"), openai.WithToken("t"),
				openai.WithModel(model), openai.WithHTTPClient(d), openai.WithModernReasoningFormat())
			if err != nil {
				t.Fatal(err)
			}
			opt := llms.WithReasoning(llms.ReasoningHigh, 0)
			if name == "off" {
				opt = llms.WithReasoningDisabled()
			}
			_, err = llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opt)
			fmt.Printf("%-32s %-4s err=%v body=%s\n", model, name, err, d.body)
		}
	}
}
