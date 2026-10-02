package openai

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestClaudeThroughAGatewayIsSentTheTemperatureRangeAnthropicDocuments(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"anthropic/claude-haiku-4-5", "claude-sonnet-4-5"} {
		for asked, sent := range map[float64]float64{1.5: 1, 1.0001: 1, -0.5: 0} {
			body, warnings := hostCall(t, "http://litellm.internal/v1", model, llms.WithTemperature(asked))
			assert.InDelta(t, sent, body["temperature"], 0, model)
			if assert.Contains(t, warnings, "WithTemperature", model) {
				assert.Equal(t, llms.WarningClamp, warnings["WithTemperature"].Kind, model)
				assert.Equal(t, strconv.FormatFloat(asked, 'g', -1, 64), warnings["WithTemperature"].Asked, model)
				assert.Equal(t, strconv.FormatFloat(sent, 'g', -1, 64), warnings["WithTemperature"].Sent, model)
			}
		}

		for _, asked := range []float64{0, 0.7, 1} {
			body, warnings := hostCall(t, "http://litellm.internal/v1", model, llms.WithTemperature(asked))
			assert.InDelta(t, asked, body["temperature"], 1e-9, model)
			assert.NotContains(t, warnings, "WithTemperature", model)
		}
	}

	body, warnings := hostCall(t, "http://litellm.internal/v1", "openai/gpt-4.1", llms.WithTemperature(1.5))
	assert.InDelta(t, 1.5, body["temperature"], 1e-9)
	assert.NotContains(t, warnings, "WithTemperature")
}

func TestAClaudeTemperatureTheDoorReplacesAnywayIsReportedOnce(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name, model string
		asked       float64
		opts        []llms.CallOption
		kind        llms.WarningKind
		sent        any
	}{
		{"a Claude that takes no sampling", "anthropic/claude-opus-4-7", 1.5, nil, llms.WarningDrop, nil},
		{"a thinking Claude above the range", "anthropic/claude-sonnet-4-5", 1.5,
			[]llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}, llms.WarningSubstitute, 1.0},
		{"a thinking Claude below the range", "anthropic/claude-sonnet-4-5", -0.5,
			[]llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}, llms.WarningSubstitute, 1.0},
	} {
		doer := &bodyDoer{}
		llm := newUnitLLM(t, WithBaseURL("http://litellm.internal/v1"), WithModel(tc.model), WithHTTPClient(doer))
		resp, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			append([]llms.CallOption{llms.WithTemperature(tc.asked)}, tc.opts...)...)
		require.NoError(t, err, tc.name)

		var body map[string]any
		require.NoError(t, json.Unmarshal(doer.body, &body), tc.name)
		if tc.sent == nil {
			assert.NotContains(t, body, "temperature", tc.name)
		} else {
			assert.InDelta(t, tc.sent, body["temperature"], 0, tc.name)
		}

		var reported []llms.Warning
		for _, w := range resp.Warnings {
			if w.Option == "WithTemperature" {
				reported = append(reported, w)
			}
		}
		if assert.Len(t, reported, 1, tc.name) {
			assert.Equal(t, tc.kind, reported[0].Kind, tc.name)
			assert.Equal(t, strconv.FormatFloat(tc.asked, 'g', -1, 64), reported[0].Asked, tc.name)
		}
	}
}
