package openai

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeBudgetLimitBody(t *testing.T) {
	body := sendModernReasoningBody(t, "anthropic/claude-opus-4-5", llms.ReasoningNone, 100, 512)
	t.Logf("body: %s", body)
}
