package openai

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeMisc(t *testing.T) {
	t.Logf("opus-4-7 off: %s", sendForWire(t, "anthropic/claude-opus-4-7", llms.WithReasoningDisabled()))
	t.Logf("reasoning-model adaptive: %s", sendForWire(t, "reasoning-model", llms.WithAdaptiveReasoning(llms.ReasoningNone)))
	t.Logf("reasoning-model high: %s", sendForWire(t, "reasoning-model", llms.WithReasoning(llms.ReasoningHigh, 0)))
	t.Logf("grok max: %s", sendForWire(t, "grok-4.5", llms.WithMaxTokens(100)))
}
