package openai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func callRecording(t *testing.T, baseURL, model string, opts ...llms.CallOption) (map[string]any, *llms.ContentResponse, error) {
	t.Helper()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	if doer.body == nil {
		return nil, resp, err
	}
	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	return body, resp, err
}

func TestALatestAliasOnTheOpenAIDoorIsRefusedWhatTheNewestReleaseOfItsTierRefuses(t *testing.T) {
	t.Parallel()

	const openRouter, liteLLM = "https://openrouter.ai/api/v1", "http://litellm.example/v1"
	for _, route := range []struct{ baseURL, model string }{
		{openRouter, "~anthropic/claude-opus-latest"},
		{openRouter, "~anthropic/claude-sonnet-latest"},
		{openRouter, "~anthropic/claude-fable-latest"},
		{liteLLM, "claude-opus-latest"},
		{liteLLM, "claude-sonnet-latest"},
		{liteLLM, "anthropic/claude-fable-latest"},
	} {
		body, _, err := callRecording(t, route.baseURL, route.model, lookupTool, llms.WithToolChoice("required"))
		var forced *reasoning.ErrForcedToolChoiceUnsupported
		require.ErrorAs(t, err, &forced, route.model)
		require.Nil(t, body, "%s: the request must not be sent", route.model)
	}

	for _, model := range []string{"claude-opus-latest", "anthropic/claude-opus-latest"} {
		body, _, err := callRecording(t, liteLLM, model, llms.WithReasoningDisabled())
		var off *reasoning.ErrReasoningOffUnsupported
		require.ErrorAs(t, err, &off, model)
		require.Nil(t, body, "%s: the request must not be sent", model)
	}

	body, resp, err := callRecording(t, liteLLM, "claude-sonnet-latest", llms.WithReasoningDisabled())
	require.NoError(t, err)
	require.Equal(t, map[string]any{"type": "between_tools"}, body["thinking"])
	var floors []llms.Warning
	for _, w := range resp.Warnings {
		if w.Option == "WithReasoningDisabled" {
			floors = append(floors, w)
		}
	}
	require.Len(t, floors, 1)
	require.Equal(t, llms.WarningSubstitute, floors[0].Kind)
	require.Equal(t, "between_tools", floors[0].Sent)
}
