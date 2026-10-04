package anthropic_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestALatestAliasIsRefusedWhatTheNewestReleaseOfItsTierRefuses(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-opus-latest", "~anthropic/claude-opus-latest", "claude-fable-latest"} {
		body, _, err := generateRecording(t, model, llms.WithReasoningDisabled())
		var off *reasoning.ErrReasoningOffUnsupported
		require.ErrorAs(t, err, &off, model)
		require.Nil(t, body, "%s: the request must not be sent", model)
	}

	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}
	for _, model := range []string{
		"claude-opus-latest", "claude-sonnet-latest", "claude-fable-latest", "~anthropic/claude-sonnet-latest",
	} {
		for _, choice := range []any{"required", llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "lookup"}}} {
			body, _, err := generateRecording(t, model, llms.WithTools([]llms.Tool{tool}), llms.WithToolChoice(choice))
			var forced *reasoning.ErrForcedToolChoiceUnsupported
			require.ErrorAs(t, err, &forced, "%s %#v", model, choice)
			require.Nil(t, body, "%s: the request must not be sent", model)
		}
	}
}

func TestTheHaikuLatestAliasIsSentOnlyOneOfTemperatureAndTopP(t *testing.T) {
	t.Parallel()

	body, resp, err := generateRecording(t, "claude-haiku-latest", llms.WithTemperature(0.5), llms.WithTopP(0.9))
	require.NoError(t, err)
	require.InDelta(t, 0.5, body["temperature"], 1e-9)
	require.NotContains(t, body, "top_p")
	var dropped bool
	for _, w := range resp.Warnings {
		dropped = dropped || w.Option == "WithTopP" && w.Kind == llms.WarningDrop
	}
	require.True(t, dropped, "%v", resp.Warnings)
}

func TestTurningThinkingOffOnTheSonnetLatestAliasSendsItsLowestSetting(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-sonnet-latest", "~anthropic/claude-sonnet-latest"} {
		body, resp, err := generateRecording(t, model, llms.WithReasoningDisabled())
		require.NoError(t, err, model)
		require.Equal(t, map[string]any{"type": "between_tools"}, body["thinking"], model)
		var floors []llms.Warning
		for _, w := range resp.Warnings {
			if w.Option == "WithReasoningDisabled" {
				floors = append(floors, w)
			}
		}
		require.Len(t, floors, 1, model)
		require.Equal(t, llms.WarningSubstitute, floors[0].Kind, model)
		require.Equal(t, "between_tools", floors[0].Sent, model)
	}
}
