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

func TestProbeForcedToolNonClaude(t *testing.T) {
	const reply = `{"id":"x","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",` +
		`"usage":{"input_tokens":1,"output_tokens":1}}`
	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "calc", Description: "multiply",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}
	for _, model := range []string{"deepseek-chat", "kimi-k2-thinking", "glm-4.6", "MiniMax-M2"} {
		var sent []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			sent, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, reply)
		}))
		llm, _ := anthropic.New(anthropic.WithToken("k"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel(model))
		opts := []llms.CallOption{llms.WithTools([]llms.Tool{tool}), llms.WithToolChoice("any"),
			llms.WithMaxTokens(8000), llms.WithReasoning(llms.ReasoningHigh, 0)}
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "17*23?")}, opts...)
		cfg := llms.CallOptions{}
		for _, o := range opts {
			o(&cfg)
		}
		shared := llms.CheckClaudeTurnLimits(model, cfg, nil)
		fmt.Printf("%-18s anthropic door err=%v sent=%v | shared CheckClaudeTurnLimits=%v\n", model, err, len(sent) > 0, shared)
		srv.Close()
	}
}
