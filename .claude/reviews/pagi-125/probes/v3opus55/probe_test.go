package reasoning_test

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestProbeOpus55(t *testing.T) {
	for _, m := range []string{"claude-opus-5-5", "anthropic.claude-opus-5-5", "us.anthropic.claude-opus-5-5", "anthropic/claude-opus-5.5", "claude-opus-5", "claude-fable-5-1", "claude-opus-latest"} {
		s := llms.ReasoningSupportFor(m, reasoning.ProviderAnthropic)
		t.Logf("%-32s alwaysOn=%v defOn=%v offAnth=%d offBedrock=%d offOpenAI=%d cannotDisable=%v",
			m, reasoning.ClaudeThinkingAlwaysOn(m), reasoning.ClaudeThinkingDefaultsOn(m),
			reasoning.ResolveOff(m, reasoning.ProviderAnthropic), reasoning.ResolveOff(m, reasoning.ProviderBedrock),
			reasoning.ResolveOff(m, reasoning.ProviderOpenAI), s.CannotDisable)
	}
}
