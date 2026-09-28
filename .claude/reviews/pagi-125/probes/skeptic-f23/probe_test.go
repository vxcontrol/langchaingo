package openai

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeSkepticF23(t *testing.T) {
	t.Log("budget body:", sendModernReasoningBody(t, "anthropic/claude-opus-4-5", llms.ReasoningNone, 100, 512))
	for _, m := range []string{"gpt-4o", "gpt-4.1", "gpt-5.1"} {
		r := sendForWarnings(t, m, llms.WithAdaptiveReasoning(llms.ReasoningNone))
		t.Logf("%s adaptive warnings=%v body=%s", m, r.Warnings, sendForWire(t, m, llms.WithAdaptiveReasoning(llms.ReasoningNone)))
		r = sendForWarnings(t, m, llms.WithReasoning(llms.ReasoningHigh, 0))
		t.Logf("%s high warnings=%v", m, r.Warnings)
	}
}
