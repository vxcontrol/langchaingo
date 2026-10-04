package openai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestAToolChoiceStaysOffTheWireWhenTheCallOffersNoTools(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-5.6-sol", "claude-sonnet-4-5", "anthropic/claude-sonnet-4-5"} {
		for _, choice := range []any{"required", "auto", "none", llms.ToolChoice{Type: "any"}} {
			body, err := wireBodyOf(t, model, nil, llms.WithToolChoice(choice), llms.WithReasoning(llms.ReasoningNone, 2048))
			require.NoError(t, err, "%s %v", model, choice)
			require.NotContains(t, body, "tool_choice", "%s %v", model, choice)
		}
	}

	body, err := wireBodyOf(t, "gpt-5.6-sol", nil, llms.WithToolChoice("required"), turnLimitTools)
	require.NoError(t, err)
	require.Equal(t, "required", body["tool_choice"])
}

func TestAForcedToolChoiceWithoutToolsIsReportedAsDropped(t *testing.T) {
	t.Parallel()

	named := llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "lookup"}}
	for _, route := range []struct{ baseURL, model string }{
		{"https://api.openai.com/v1", "gpt-4.1"},
		{"http://litellm.internal/v1", "anthropic/claude-sonnet-4-5"},
	} {
		for _, tc := range []struct {
			choice any
			asked  string
		}{
			{"required", "any"},
			{llms.ToolChoice{Type: "any"}, "any"},
			{named, "lookup"},
		} {
			body, warnings := hostCall(t, route.baseURL, route.model, llms.WithToolChoice(tc.choice))
			require.NotContains(t, body, "tool_choice", "%s %v", route.model, tc.choice)
			w, reported := warnings["WithToolChoice"]
			require.True(t, reported, "%s %v: %v", route.model, tc.choice, warnings)
			require.Equal(t, llms.WarningDrop, w.Kind)
			require.Equal(t, tc.asked, w.Asked)
			require.Empty(t, w.Sent)
		}

		for _, choice := range []any{"auto", "none"} {
			_, warnings := hostCall(t, route.baseURL, route.model, llms.WithToolChoice(choice))
			require.NotContains(t, warnings, "WithToolChoice", "%s %v: the vendor's default without tools", route.model, choice)
		}
	}
}

func TestToolsInTheExtraBodyKeepAForcedChoiceOnTheWire(t *testing.T) {
	t.Parallel()

	extraTools := llms.WithExtraBody(map[string]any{"tools": []any{
		map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}},
	}})
	required := llms.WithToolChoice("required")

	body, warnings := hostCall(t, "https://api.openai.com/v1", "gpt-4.1", extraTools, required)
	require.Equal(t, "required", body["tool_choice"])
	require.NotEmpty(t, body["tools"])
	require.NotContains(t, warnings, "WithToolChoice")

	for _, route := range []struct{ baseURL, model string }{
		{"http://litellm.internal/v1", "anthropic/claude-opus-5-5"},
		{"https://api.moonshot.ai/v1", "kimi-k2.6"},
	} {
		doer := &bodyDoer{}
		llm := newUnitLLM(t, WithBaseURL(route.baseURL), WithModel(route.model), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "look it up")}, extraTools, required)
		var refused *reasoning.ErrForcedToolChoiceUnsupported
		require.ErrorAs(t, err, &refused, route.model)
		require.Nil(t, doer.body, "%s: refused before the network", route.model)
	}
}

func TestFunctionsTheOpenAIDoorSendsAsToolsKeepAForcedChoiceRefusedOnClaude55(t *testing.T) {
	t.Parallel()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL("http://litellm.internal/v1"), WithModel("anthropic/claude-opus-5-5"),
		WithHTTPClient(doer))
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time is it?")},
		llms.WithFunctions([]llms.FunctionDefinition{{Name: "now"}}), llms.WithToolChoice("required"))
	var refused *reasoning.ErrForcedToolChoiceUnsupported
	require.ErrorAs(t, err, &refused)
	require.Nil(t, doer.body)
}
