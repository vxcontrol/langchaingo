package reasoning_test

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestProbeSig(t *testing.T) {
	t.Log(reasoning.ClaudeSupportsEffortWithBudget("claude-opus-4-5"))
}
