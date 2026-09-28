package llms

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestProbeTrcStop(t *testing.T) {
	resp := &ContentResponse{Choices: []*ContentChoice{{Content: "cut", StopReason: "model_context_window_exceeded"}}}
	t.Logf("IsTruncated(model_context_window_exceeded)=%v CheckTruncation(fail)=%v",
		IsTruncated("model_context_window_exceeded"), CheckTruncation(resp, CallOptions{FailOnTruncation: true}))
}

func TestProbeTrcHint(t *testing.T) {
	for _, m := range []string{"deepseek-v3.1", "qwen3-max", "kimi-k3", "kimi-k2.6", "kimi-k2.7-code", "deepseek/deepseek-v4-pro"} {
		s := ReasoningSupportFor(m, reasoning.ProviderOpenAI)
		t.Logf("%s: IsReasoning=%v capsKnown=%v RejectsSampling(hint)=%v RejectsSamplingWhileThinking=%v FixesSampling=%v",
			m, reasoning.IsReasoningModel(m), reasoning.OpenAIReasoningCapsFor(m).Known, s.RejectsSampling,
			reasoning.RejectsSamplingWhileThinking(m), reasoning.FixesSampling(m))
	}
}
