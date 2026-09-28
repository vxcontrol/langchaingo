package reasoning

import (
	"fmt"
	"testing"
)

func TestProbeDump(t *testing.T) {
	models := []string{
		"claude-opus-5-5", "anthropic.claude-opus-5-5", "us.anthropic.claude-opus-5-5", "anthropic/claude-opus-5-5",
		"claude-opus-5", "claude-opus-latest", "claude-fable-5-1", "claude-mythos-5-1",
		"anthropic/claude-sonnet-4", "anthropic/claude-opus-4", "claude-sonnet-4-0", "claude-sonnet-4-20250514",
		"anthropic/claude-3.7-sonnet", "anthropic/claude-3.7-sonnet:thinking",
	}
	for _, m := range models {
		fmt.Printf("%-40s spell=%v kind=%d alwaysOn=%v defOn=%v offA=%d offB=%d offO=%d predates=%v so=%v interleave=%v rejS=%v prefill=%v isR=%v\n",
			m, modelSpellings(m), ClaudeReasoningKindFor(m), ClaudeThinkingAlwaysOn(m), ClaudeThinkingDefaultsOn(m),
			ResolveOff(m, ProviderAnthropic), ResolveOff(m, ProviderBedrock), ResolveOff(m, ProviderOpenAI),
			ClaudePredatesAdaptive(m), ClaudeSupportsStructuredOutput(m), ClaudeInterleavesOnBudget(m),
			ClaudeRejectsSampling(m), ClaudeRejectsAssistantPrefill(m), IsReasoningModel(m))
	}
}
