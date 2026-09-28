package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func wireBodyOf(t *testing.T, model string, opts []Option, call ...llms.CallOption) (map[string]any, error) {
	t.Helper()

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"`+model+`",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)

	llm := newUnitLLM(t, append([]Option{WithBaseURL(srv.URL), WithModel(model)}, opts...)...)

	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, call...)
	if err != nil {
		return nil, err
	}

	var body map[string]any
	require.NoError(t, json.Unmarshal(raw, &body))
	return body, nil
}

// offAndAdaptive reaches past the call options: each of them replaces the whole
// config, so only a caller filling the exported struct can set both.
func offAndAdaptive() llms.CallOption {
	return func(o *llms.CallOptions) {
		o.Reasoning = &llms.ReasoningConfig{Mode: llms.ReasoningOff, Adaptive: true}
	}
}

func TestAnExplicitOffOutranksTheAdaptiveFlag(t *testing.T) {
	t.Parallel()

	body, err := wireBodyOf(t, "gpt-5.1", nil, offAndAdaptive())
	require.NoError(t, err)

	require.Contains(t, body, "reasoning_effort",
		"the caller asked to switch thinking off, and the vendor needs the field to hear it")
	assert.Equal(t, "none", body["reasoning_effort"])
}

func TestAdaptiveAloneStillLeavesTheDepthToTheVendor(t *testing.T) {
	t.Parallel()

	body, err := wireBodyOf(t, "gpt-5.1", nil, llms.WithAdaptiveReasoning(""))
	require.NoError(t, err)

	assert.NotContains(t, body, "reasoning_effort",
		"an adaptive request with no effort hands the depth to the vendor")
}

func TestABudgetRequestWithToolsReachesTheVendor(t *testing.T) {
	t.Parallel()

	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name:       "lookup",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}

	body, err := wireBodyOf(t, "gpt-5.6", []Option{WithModernReasoningFormat(), WithUsingReasoningMaxTokens()},
		llms.WithTools([]llms.Tool{tool}), llms.WithReasoning(llms.ReasoningNone, 3000))
	require.NoError(t, err, "a budget is not an effort; tools do not forbid it")

	reasoningField, ok := body["reasoning"].(map[string]any)
	require.True(t, ok, "the budget must reach the wire: %v", body)
	assert.InDelta(t, 3000, reasoningField["max_tokens"], 0)
	assert.NotContains(t, reasoningField, "effort",
		"the effort the vendor refuses next to tools stays off the wire")
}

func TestAnEffortRequestWithToolsIsStillRefusedBeforeTheNetwork(t *testing.T) {
	t.Parallel()

	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name:       "lookup",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}

	_, err := wireBodyOf(t, "gpt-5.6", nil,
		llms.WithTools([]llms.Tool{tool}), llms.WithReasoning(llms.ReasoningHigh, 0))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "rejects reasoning effort")
}
