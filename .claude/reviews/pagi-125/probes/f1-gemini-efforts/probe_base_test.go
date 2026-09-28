package llms_test

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestProbeGeminiEfforts(t *testing.T) {
	for _, m := range []string{"gemini-3-pro-preview", "gemini-3-flash-preview"} {
		s := llms.ReasoningSupportFor(m, reasoning.ProviderGoogleAI)
		t.Logf("%s Known=%v Efforts=%v", m, s.Known, s.Efforts)
	}
}
