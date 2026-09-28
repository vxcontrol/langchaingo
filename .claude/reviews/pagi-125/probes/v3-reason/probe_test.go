package llms_test

import (
	"fmt"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestProbeV3(t *testing.T) {
	for _, m := range []string{"deepseek-ai/DeepSeek-V3", "deepseek-chat", "deepseek-ai/DeepSeek-R1", "grok-4", "gpt-4o"} {
		fmt.Printf("RejectsPenalties(%q)=%v\n", m, reasoning.RejectsPenalties(m))
	}
	for _, m := range []string{"claude-opus-latest", "claude-sonnet-latest", "claude-fable-latest", "claude-opus-4-7", "claude-opus-5"} {
		fmt.Printf("prefill %q rejects=%v kind=%v alwaysOn=%v\n", m, reasoning.ClaudeRejectsAssistantPrefill(m), reasoning.ClaudeReasoningKindFor(m), reasoning.ClaudeThinkingAlwaysOn(m))
	}
	for _, m := range []string{"openai.gpt-5.6-luna", "openai.gpt-5.2", "moonshotai.kimi-k3", "gpt-5.2", "openai.gpt-oss-120b-1:0"} {
		for _, p := range []reasoning.Provider{reasoning.ProviderBedrock, reasoning.ProviderOllama, reasoning.ProviderGoogleAI, reasoning.ProviderOpenAI} {
			s := llms.ReasoningSupportFor(m, p)
			fmt.Printf("support %q p=%v Known=%v Supported=%v Efforts=%v Mech=%v | ResolveMechanism(bedrock-ish)=%v\n", m, p, s.Known, s.Supported, s.Efforts, s.Mechanism, reasoning.ResolveMechanism(m, false, false, reasoning.IsReasoningModel(m)))
		}
	}
}
