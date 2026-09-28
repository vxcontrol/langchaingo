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
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestProbeOpus55Off(t *testing.T) {
	const resp = `{"id":"x","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",` +
		`"usage":{"input_tokens":1,"output_tokens":1}}`
	for _, model := range []string{"claude-opus-5-5", "claude-fable-5-1", "claude-opus-5"} {
		var body string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			body = string(b)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, resp)
		}))
		llm, _ := anthropic.New(anthropic.WithBaseURL(srv.URL), anthropic.WithToken("t"), anthropic.WithModel(model))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithReasoningDisabled())
		srv.Close()
		s := llms.ReasoningSupportFor(model, reasoning.ProviderAnthropic)
		fmt.Printf("OFF %-18s err=%v cannotDisable=%v body=%s\n", model, err, s.CannotDisable, body)
	}
}
