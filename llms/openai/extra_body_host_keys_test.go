package openai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

const litellmHost = "http://litellm.internal:4000/v1"

func TestTheExtraBodyThinkingSwitchCountsOnlyTheKeysTheHostReads(t *testing.T) {
	t.Parallel()

	const (
		openAI   = "https://api.openai.com/v1"
		deepSeek = "https://api.deepseek.com"
	)
	high := []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}
	for name, tc := range map[string]struct {
		baseURL, model string
		extra          map[string]any
		thinks         bool
		opts           []llms.CallOption
	}{
		"OpenAI does not read enable_thinking":                  {openAI, "gpt-5.4", map[string]any{"enable_thinking": false}, true, high},
		"OpenAI does not switch thinking on by enable_thinking": {openAI, "gpt-5.4", map[string]any{"enable_thinking": true}, false, nil},
		"a model that always thinks is not switched off":        {openAI, "gpt-5", map[string]any{"reasoning_effort": "none"}, true, nil},
		"an effort typed as the library's own string":           {openAI, "gpt-5.4", map[string]any{"reasoning_effort": llms.ReasoningHigh}, true, nil},
		"DeepSeek does not read enable_thinking":                {deepSeek, "deepseek-v4-pro", map[string]any{"enable_thinking": false}, true, nil},
		"DeepSeek does not read the reasoning object":           {deepSeek, "deepseek-v4-pro", map[string]any{"reasoning": map[string]any{"enabled": false}}, true, nil},
		"a thinking object typed as a string map":               {deepSeek, "deepseek-v4-pro", map[string]any{"thinking": map[string]string{"type": "disabled"}}, false, nil},
		"an unknown host reads every key, a budget in int64":    {litellmHost, "anthropic/claude-sonnet-4-5", map[string]any{"reasoning": map[string]any{"max_tokens": int64(2048)}}, true, nil},
	} {
		body, _ := hostCall(t, tc.baseURL, tc.model,
			append([]llms.CallOption{llms.WithTemperature(0.7), llms.WithExtraBody(tc.extra)}, tc.opts...)...)
		temperature, sent := body["temperature"]
		if tc.thinks {
			assert.True(t, !sent || temperature == float64(1), "%s: a thinking request keeps no caller temperature: %v", name, body)
			continue
		}
		assert.Equal(t, 0.7, temperature, "%s: %v", name, body)
	}
}

func TestAForcedChoiceIsReadAsTheMergedExtraBodySendsIt(t *testing.T) {
	t.Parallel()

	lookup := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}})
	call := func(baseURL, model string, opts ...llms.CallOption) error {
		doer := &bodyDoer{}
		llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
		if err == nil {
			require.NotNil(t, doer.body)
		} else {
			require.Nil(t, doer.body, "a refusal comes before the request")
		}
		return err
	}

	var forced *reasoning.ErrForcedToolChoiceUnsupported
	require.ErrorAs(t, call("https://api.z.ai/api/paas/v4", "glm-5.1", llms.WithExtraBody(map[string]any{
		"tools":       []any{map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}}},
		"tool_choice": "required",
	})), &forced, "Z.ai takes only auto")
	require.ErrorAs(t, call("https://api.deepseek.com", "deepseek-v4-pro", lookup,
		llms.WithToolChoice(llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "lookup"}}),
		llms.WithExtraBody(map[string]any{"tool_choice": map[string]any{"function": map[string]any{"name": "lookup"}}})),
		&forced, "the partial object merges into a named choice DeepSeek refuses while thinking")
	require.ErrorAs(t, call("https://api.deepseek.com", "deepseek-v4-pro", lookup,
		llms.WithExtraBody(map[string]any{"tool_choice": json.RawMessage(`"required"`)})),
		&forced, "a choice spelled as raw JSON")

	var withThinking *reasoning.ErrForcedToolUseWithThinking
	require.ErrorAs(t, call(litellmHost, "anthropic/claude-sonnet-4-5", lookup, llms.WithToolChoice("required"),
		llms.WithExtraBody(map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": 2048}})),
		&withThinking, "Claude's thinking switched on in the extra body")
	require.ErrorAs(t, call(litellmHost, "anthropic/claude-sonnet-4-5", lookup, llms.WithReasoning(llms.ReasoningHigh, 2048),
		llms.WithExtraBody(map[string]any{"tool_choice": "required"})),
		&withThinking, "the forced choice set in the extra body")
	require.NoError(t, call(litellmHost, "anthropic/claude-sonnet-4-5", lookup, llms.WithToolChoice("required"),
		llms.WithExtraBody(map[string]any{"thinking": map[string]any{"type": "adaptive"}})))
}

func TestTheDefaultHostReadsTheExtraBodyAsOpenAIDoes(t *testing.T) {
	t.Parallel()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithModel("gpt-5.4"), WithHTTPClient(doer))
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithTemperature(0.7),
		llms.WithExtraBody(map[string]any{"enable_thinking": false}))
	require.NoError(t, err)
	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	assert.NotContains(t, body, "temperature", "OpenAI does not read enable_thinking, so the effort still thinks: %v", body)
}
