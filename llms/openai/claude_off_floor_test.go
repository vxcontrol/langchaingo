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

func claudeThroughAGateway(t *testing.T, model string, opts ...llms.CallOption) (*llms.ContentResponse, map[string]any) {
	t.Helper()

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
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	require.NoError(t, err, model)

	var body map[string]any
	require.NoError(t, json.Unmarshal(raw, &body), model)
	return resp, body
}

func TestTurningThinkingOffOnClaudeSonnet55ThroughAGatewaySendsItsLowestSetting(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"claude-sonnet-5-5", "anthropic/claude-sonnet-5-5", "bedrock/us.anthropic.claude-sonnet-5-5",
		"vertex_ai/claude-sonnet-5-5",
	} {
		resp, body := claudeThroughAGateway(t, model, llms.WithReasoningDisabled())
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

func TestClaudeSonnet55TurnedOffThroughAGatewaySendsNoEffortBesideItsLowestSetting(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"} {
		_, body := claudeThroughAGateway(t, model, func(o *llms.CallOptions) {
			o.Reasoning = &llms.ReasoningConfig{Mode: llms.ReasoningOff, Effort: llms.ReasoningXHigh}
		})
		require.Equal(t, map[string]any{"type": "between_tools"}, body["thinking"], model)
		require.NotContains(t, body, "reasoning_effort", model)
		require.NotContains(t, body, "output_config", model)
	}
}

func TestClaudeOpus41ThroughAGatewayIsSentOnlyOneOfTemperatureAndTopP(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"anthropic/claude-opus-4-1", "bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0", "claude-opus-4.1",
	} {
		_, body := claudeThroughAGateway(t, model, llms.WithTemperature(0.5), llms.WithTopP(0.9))
		require.InDelta(t, 0.5, body["temperature"], 1e-9, model)
		require.NotContains(t, body, "top_p", model)
	}
}
