package anthropic_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
)

func TestProbeV3NonClaude(t *testing.T) {
	for _, m := range []string{"deepseek-chat", "kimi-k2-thinking", "glm-4.6", "MiniMax-M2", "claude-sonnet-4-5"} {
		var body string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			body = string(b)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"m","type":"message","role":"assistant","model":"x","content":[{"type":"tool_use","id":"t1","name":"f","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		llm, _ := anthropic.New(anthropic.WithToken("k"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel(m))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithReasoning(llms.ReasoningHigh, 0),
			llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{Name: "f", Parameters: map[string]any{"type": "object"}}}}),
			llms.WithToolChoice("any"))
		srv.Close()
		fmt.Printf("model=%s err=%v sent=%v body=%s\n", m, err, body != "", body)
	}
}
