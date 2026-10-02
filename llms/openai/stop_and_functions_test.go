package openai

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestStopWordsAreRefusedBeforeTheNetworkWhereTheVendorRefusesThem(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ baseURL, model string }{
		{"https://api.openai.com/v1", "o3"},
		{"https://api.openai.com/v1", "o3-2025-04-16"},
		{"https://api.openai.com/v1", "o4-mini"},
		{"https://api.openai.com/v1", "o4-mini-2025-04-16"},
		{"https://api.x.ai/v1", "grok-4.7"},
		{"https://api.x.ai/v1", "grok-4.6"},
	} {
		doer := &bodyDoer{}
		llm := newUnitLLM(t, WithBaseURL(tc.baseURL), WithModel(tc.model), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithStopWords([]string{"END"}))
		var refused *reasoning.ErrStopWordsUnsupported
		require.True(t, errors.As(err, &refused), "%s: %v", tc.model, err)
		require.Nil(t, doer.body, "%s: refused before the network", tc.model)
	}

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL("http://litellm.internal/v1"), WithModel("xai/grok-4.7"), WithHTTPClient(doer))
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithStopWords([]string{"END"}))
	var refused *reasoning.ErrStopWordsUnsupported
	require.True(t, errors.As(err, &refused), "the gateway's xai route is xAI's API: %v", err)

	for _, tc := range []struct{ baseURL, model string }{
		{"https://api.openai.com/v1", "gpt-4.1"},
		{"https://api.openai.com/v1", "o3-mini"},
		{"https://api.x.ai/v1", "grok-4-1-fast-non-reasoning"},
		{"https://api.x.ai/v1", "grok-3"},
		{"https://openrouter.ai/api/v1", "x-ai/grok-4.7"},
		{"https://openrouter.ai/api/v1", "openai/o3"},
	} {
		body, _ := hostCall(t, tc.baseURL, tc.model, llms.WithStopWords([]string{"END"}))
		assert.Equal(t, []any{"END"}, body["stop"], tc.model)
	}
}

func TestFunctionsFollowTheEffortRuleForTools(t *testing.T) {
	t.Parallel()

	functions := llms.WithFunctions([]llms.FunctionDefinition{{
		Name: "lookup", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}})

	_, err := wireBodyOf(t, "gpt-5.4", nil, functions, llms.WithReasoning(llms.ReasoningHigh, 0))
	var withTools *reasoning.ErrEffortWithTools
	require.True(t, errors.As(err, &withTools), "%v", err)

	body, err := wireBodyOf(t, "gpt-5.6", nil, functions)
	require.NoError(t, err)
	assert.Equal(t, reasoning.OpenAIDisableEffort, body["reasoning_effort"])
}
