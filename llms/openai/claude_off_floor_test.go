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

	for _, model := range []string{
		"claude-sonnet-5-5", "anthropic/claude-sonnet-5-5", "bedrock/us.anthropic.claude-sonnet-5-5",
		"vertex_ai/claude-sonnet-5-5",
	} {
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
		require.NoError(t, err, model)

		var body map[string]any
		require.NoError(t, json.Unmarshal(raw, &body), model)
		require.Equal(t, map[string]any{"type": "between_tools"}, body["thinking"], model)
		var floors []llms.Warning
		for _, w := range resp.Warnings {
			if w.Option == "WithReasoningDisabled" {
				floors = append(floors, w)
			}
		}
		require.Len(t, floors, 1, model)
		require.Equal(t, llms.WarningSubstitute, floors[0].Kind, model)
		require.Equal(t, "off", floors[0].Asked, model)
		require.Equal(t, "between_tools", floors[0].Sent, model)
	}
}
