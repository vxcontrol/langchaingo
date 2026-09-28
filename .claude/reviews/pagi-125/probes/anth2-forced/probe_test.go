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

func TestProbeForcedToolOnNewModels(t *testing.T) {
	const reply = `{"id":"x","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",` +
		`"usage":{"input_tokens":1,"output_tokens":1}}`
	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "calc", Description: "multiply",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}
	for _, model := range []string{"claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1", "claude-sonnet-4-5"} {
		for name, extra := range map[string][]llms.CallOption{
			"default":   nil,
			"reasoning": {llms.WithReasoning(llms.ReasoningHigh, 0)},
		} {
			var sent []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				sent, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, reply)
			}))
			llm, _ := anthropic.New(anthropic.WithToken("k"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel(model))
			opts := append([]llms.CallOption{llms.WithTools([]llms.Tool{tool}), llms.WithToolChoice("any"), llms.WithMaxTokens(8000)}, extra...)
			_, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "17*23?")}, opts...)
			fmt.Printf("%-18s %-9s err=%v sentRequest=%v\n", model, name, err, len(sent) > 0)
			srv.Close()
		}
	}
}
