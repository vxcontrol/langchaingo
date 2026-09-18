package anthropic_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAnEffortThatTravelledAsTheBudgetIsCalledASubstitution(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithMaxTokens(8192), llms.WithReasoning(llms.ReasoningMedium, 0))

	found := warningsFor(resp.Warnings, "WithReasoning")
	require.Len(t, found, 1, "one warning for one ask: %v", resp.Warnings)
	require.Equal(t, llms.WarningSubstitute, found[0].Kind)
	require.Equal(t, "medium", found[0].Asked)
	require.Contains(t, found[0].Sent, "tokens")
}

func TestAnEffortAnExplicitBudgetOverrodeIsCalledALoss(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithMaxTokens(8192), llms.WithReasoning(llms.ReasoningMedium, 2048))

	var dropped *llms.Warning
	for _, w := range warningsFor(resp.Warnings, "WithReasoning") {
		if w.Kind == llms.WarningDrop {
			dropped = &w
		}
	}
	require.NotNil(t, dropped, "the effort went nowhere unreported: %v", resp.Warnings)
	require.Equal(t, "medium", dropped.Asked)
}
