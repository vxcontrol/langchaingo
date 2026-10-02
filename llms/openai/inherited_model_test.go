package openai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestAnUnlistedVersionIsSentWhatItsReleaseWouldRefuse(t *testing.T) {
	t.Parallel()

	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}})
	inherited := func(warnings map[string]llms.Warning, option string) {
		t.Helper()
		require.Equal(t, llms.WarningInherit, warnings[option].Kind, "%s: %v", option, warnings)
	}

	body, warnings := hostCall(t, "https://api.openai.com/v1", "gpt-6.2-sol", tools)
	require.Len(t, body["tools"], 1)
	inherited(warnings, "WithTools")

	body, warnings = hostCall(t, "https://api.x.ai/v1", "grok-4.8", llms.WithStopWords([]string{"END"}))
	require.Equal(t, []any{"END"}, body["stop"])
	inherited(warnings, "WithStopWords")

	body, warnings = hostCall(t, "https://api.x.ai/v1", "grok-4.8", llms.WithReasoningDisabled())
	require.Equal(t, "none", body["reasoning_effort"], "grok-4.3 is the newest grok that documents none")
	inherited(warnings, "WithReasoningDisabled")

	body, _ = hostCall(t, "https://api.z.ai/api/paas/v4", "glm-6", llms.WithReasoningDisabled())
	thinking, _ := body["thinking"].(map[string]any)
	require.Equal(t, "disabled", thinking["type"], "glm-5.2 is the newest GLM that documents a disable: %v", body)

	body, _ = hostCall(t, "https://api.mistral.ai/v1", "zai-glm-6", llms.WithReasoningDisabled())
	require.NotContains(t, body, "thinking", "Mistral's GLM releases turn thinking off by omission: %v", body)

	body, warnings = hostCall(t, "https://api.openai.com/v1", "gpt-5.7", tools, llms.WithReasoning(llms.ReasoningHigh, 0))
	require.Equal(t, "high", body["reasoning_effort"])
	require.Len(t, body["tools"], 1)
	inherited(warnings, "WithReasoning")

	body, warnings = hostCall(t, "https://api.deepseek.com", "deepseek-v5", tools,
		llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithToolChoice("required"))
	require.Equal(t, "required", body["tool_choice"])
	inherited(warnings, "WithToolChoice")

	_, warnings = hostCall(t, "https://api.openai.com/v1", "gpt-5.7-cyber")
	inherited(warnings, "WithModel")
}

func TestAListedReleaseKeepsItsRefusals(t *testing.T) {
	t.Parallel()

	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}})
	call := func(baseURL, model string, opts ...llms.CallOption) error {
		llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(&bodyDoer{}))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
		return err
	}

	var chatTools *reasoning.ErrChatToolsUnsupported
	require.ErrorAs(t, call("https://api.openai.com/v1", "gpt-6.1-sol", tools), &chatTools)
	var stop *reasoning.ErrStopWordsUnsupported
	require.ErrorAs(t, call("https://api.x.ai/v1", "grok-4.7", llms.WithStopWords([]string{"END"})), &stop)
	var off *reasoning.ErrReasoningOffUnsupported
	require.ErrorAs(t, call("https://api.x.ai/v1", "grok-4.7", llms.WithReasoningDisabled()), &off)
	require.ErrorAs(t, call("https://api.z.ai/api/paas/v4", "glm-5.3", llms.WithReasoningDisabled()), &off)
	require.ErrorAs(t, call("https://api.minimax.io/v1", "MiniMax-M2.7", llms.WithReasoningDisabled()), &off)
	var cyber *reasoning.ErrChatCompletionsUnsupported
	require.ErrorAs(t, call("https://api.openai.com/v1", "gpt-5.6-cyber"), &cyber)
}
