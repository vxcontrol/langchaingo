package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestTurningThinkingOffOnClaudeSonnet55ThroughAGatewaySendsItsLowestSetting(t *testing.T) {
	t.Parallel()

	const model = "claude-sonnet-5-5"
	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"`+model+`",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)

	llm := newUnitLLM(t, WithBaseURL(srv.URL), WithModel(model))
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithReasoningDisabled())
	require.NoError(t, err)

	var body map[string]any
	require.NoError(t, json.Unmarshal(raw, &body))
	require.Equal(t, map[string]any{"type": "between_tools"}, body["thinking"])
	var floors []llms.Warning
	for _, w := range resp.Warnings {
		if w.Option == "WithReasoningDisabled" {
			floors = append(floors, w)
		}
	}
	require.Len(t, floors, 1)
	require.Equal(t, "between_tools", floors[0].Sent)
}
