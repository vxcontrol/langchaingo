package openai

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestProbeV3Hint(t *testing.T) {
	for _, m := range []string{"anthropic/claude-sonnet-5", "anthropic/claude-opus-5", "claude-sonnet-5"} {
		hint := llms.ReasoningSupportFor(m, reasoning.ProviderOpenAI)
		for _, u := range []string{"http://openrouter.ai/api/v1", "http://api.anthropic.com/v1", "http://litellm.local:4000"} {
			sent, err := sendOffToHost(t, u, m, nil)
			t.Logf("model=%s host=%s hint.CannotDisable=%v sent=%v err=%v", m, u, hint.CannotDisable, sent, err)
		}
	}
}
