package llms_test

import (
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"testing"
)

func TestProbeV3Samp(t *testing.T) {
	for _, m := range []string{"kimi-k3", "kimi-k2.6", "kimi-k2.7-code", "moonshotai/kimi-k3", "deepseek-v3.1", "qwen3-max"} {
		s := llms.ReasoningSupportFor(m, reasoning.ProviderOpenAI)
		t.Logf("%s: hint=%+v RejectsSamplingWhileThinking=%v FixesSampling=%v", m, s, reasoning.RejectsSamplingWhileThinking(m), reasoning.FixesSampling(m))
	}
}
