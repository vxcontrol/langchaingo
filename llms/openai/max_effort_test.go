package openai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func assertMaxSentAsXHigh(t *testing.T, body map[string]any, warnings map[string]llms.Warning, why string) {
	t.Helper()

	assert.Equal(t, "xhigh", body["reasoning_effort"], why)
	if assert.Contains(t, warnings, "WithReasoning", why) {
		assert.Equal(t, llms.WarningClamp, warnings["WithReasoning"].Kind, why)
		assert.Equal(t, "max", warnings["WithReasoning"].Asked, why)
		assert.Equal(t, "xhigh", warnings["WithReasoning"].Sent, why)
	}
}

func TestOpenAIsMaxGoesToTheResponsesAPIWhereverItCan(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol",
	} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			for _, baseURL := range []string{"", "https://llm.pentagi.net/openai_passthrough/v1"} {
				body, err := callWithATool(t, baseURL, model, llms.WithReasoning(llms.ReasoningMax, 0))
				require.NoError(t, err, baseURL)
				assert.Equal(t, map[string]any{"effort": "max"}, body["reasoning"], "a call with tools on %q", baseURL)

				body, warnings := hostCall(t, baseURL, model, llms.WithReasoning(llms.ReasoningMax, 0))
				assert.Contains(t, body, "input", "a call without tools goes to the Responses API on %q", baseURL)
				assert.Equal(t, map[string]any{"effort": "max"}, body["reasoning"], "a call without tools on %q", baseURL)
				assert.NotContains(t, warnings, "WithReasoning", baseURL)

				body, _ = hostCall(t, baseURL, model, llms.WithReasoning(llms.ReasoningXHigh, 0))
				assert.Equal(t, "xhigh", body["reasoning_effort"], "an effort Chat Completions takes stays there on %q", baseURL)
			}

			doer := &bodyDoer{}
			_, err := newUnitLLM(t, WithModel(model), WithHTTPClient(doer)).GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
				llms.WithReasoning(llms.ReasoningMax, 0), llms.WithStopWords([]string{"END"}))
			var stop *reasoning.ErrStopWordsUnsupported
			require.ErrorAs(t, err, &stop, "Chat Completions answers 400 to stop on this model and the Responses API has no stop")
			assert.Nil(t, doer.body, "the refusal comes before the network")

			body, warnings := hostCall(t, "http://litellm.internal/v1", "openai/"+model, llms.WithReasoning(llms.ReasoningMax, 0))
			assertMaxSentAsXHigh(t, body, warnings, "a gateway route serves OpenAI's Chat Completions only")
		})
	}
}

func TestToolsOnlyInTheExtraBodyKeepTheChatShapeWhenMaxIsAsked(t *testing.T) {
	t.Parallel()

	toolsOnlyOnResponses := map[string]bool{"gpt-6-astra": true, "gpt-6.1-sol": true}
	for _, model := range []string{
		"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol",
	} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			doer := &bodyDoer{}
			_, err := newUnitLLM(t, WithModel(model), WithHTTPClient(doer)).GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "look it up")},
				llms.WithExtraBody(map[string]any{"tools": []any{
					map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}},
				}}),
				llms.WithReasoning(llms.ReasoningMax, 0))
			if toolsOnlyOnResponses[model] {
				var unsupported *reasoning.ErrChatToolsUnsupported
				require.ErrorAs(t, err, &unsupported)
			} else {
				var withTools *reasoning.ErrEffortWithTools
				require.ErrorAs(t, err, &withTools)
			}
			assert.Nil(t, doer.body, "the refusal comes before the network")
		})
	}
}
