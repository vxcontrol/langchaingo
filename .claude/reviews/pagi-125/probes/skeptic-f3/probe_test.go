package llms_test

import (
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"testing"
)

func TestProbeF3(t *testing.T) {
	t.Logf("%+v", llms.ReasoningSupportFor("openai.gpt-5.6-luna", reasoning.ProviderBedrock))
	t.Logf("mech=%v gptoss=%v", reasoning.ResolveMechanism("openai.gpt-5.6-luna", false, false, true), reasoning.IsGptOssModel("openai.gpt-5.6-luna"))
	t.Logf("rp=%v %v", reasoning.RejectsPenalties("deepseek-ai/DeepSeek-V3"), reasoning.ClaudeRejectsAssistantPrefill("anthropic/claude-opus-latest"))
}
