package reasoning

import (
	"fmt"
	"testing"
)

func TestProbeR2Tables(t *testing.T) {
	for _, m := range []string{"anthropic/claude-sonnet-4", "anthropic/claude-opus-4", "anthropic/claude-sonnet-4.5",
		"anthropic/claude-opus-4.1", "claude-sonnet-4-20250514", "claude-sonnet-4-0", "claude-opus-4@20250514"} {
		fmt.Printf("CLAUDE %-30s kind=%d thinks=%v predates=%v interleaves=%v SO=%v adaptive(true)=%v isR=%v\n", m,
			ClaudeReasoningKindFor(m), ClaudeSupportsThinking(m), ClaudePredatesAdaptive(m), ClaudeInterleavesOnBudget(m),
			ClaudeSupportsStructuredOutput(m), ResolveClaudeAdaptive(m, true), IsReasoningModel(m))
	}
	for _, m := range []string{"glm-latest", "glm-5.3", "glm-flash-latest", "glm-5.3-flash"} {
		c := OpenAIReasoningCapsFor(m)
		fmt.Printf("GLM %-18s mandatory=%v caps=%v/%v medium->%q xhigh->%q minimal->%q\n", m, mandatoryThinking(m), c.Known, c.Efforts,
			c.ClampEffort("medium"), c.ClampEffort("xhigh"), c.ClampEffort("minimal"))
	}
	for _, m := range []string{"openai/gpt-5.4", "openai/gpt-5.6", "gpt-5.4"} {
		fmt.Printf("TOOLS %-16s rule=%d\n", m, EffortWithTools(m))
	}
	fmt.Printf("MINIMAX TakesNoResponseFormat(minimax/minimax-m2)=%v TakesNoTopK=%v ThinkTags=%v\n",
		TakesNoResponseFormat("minimax/minimax-m2"), TakesNoTopK("minimax/minimax-m2"), ReplaysReasoningInThinkTags("minimax/minimax-m2"))
	fmt.Printf("DEEPSEEK ServedByDeepSeek(deepseek/deepseek-v4-flash, openrouter.ai)=%v\n", ServedByDeepSeek("deepseek/deepseek-v4-flash", "openrouter.ai"))
	for _, m := range []string{"gemini-3.1-flash-tts-preview", "gemini-3.1-flash-image", "gemini-2.5-flash-image", "gemini-2.5-flash-preview-tts"} {
		fmt.Printf("GEMINI %-30s thinks=%v off=%d (minimal=%d zero=%d omit=%d unsup=%d)\n", m, GeminiSupportsThinking(m), ResolveOff(m, ProviderGoogleAI), OffMinimalLevel, OffZeroBudget, OffOmit, OffUnsupported)
	}
}
