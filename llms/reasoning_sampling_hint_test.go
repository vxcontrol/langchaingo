package llms_test

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"

	"github.com/stretchr/testify/assert"
)

func TestTheHintReportsTheSamplingTheWireRefuses(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"o3-mini", "gpt-5-mini", "o3", "gpt-5.1"} {
		support := llms.ReasoningSupportFor(model, reasoning.ProviderOpenAI)
		assert.True(t, support.Supported, "%s is a reasoning model", model)
		assert.Equal(t, reasoning.RejectsSamplingWhileThinking(model), support.RejectsSampling,
			"%s: the hint and the wire disagree about sampling", model)
		assert.True(t, support.RejectsSampling,
			"%s pins temperature and drops top_p on the wire", model)
	}
}

func TestAForeignVendorOnTheOpenAIDoorKeepsItsSampling(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"deepseek-v3.1", "qwen3-max"} {
		support := llms.ReasoningSupportFor(model, reasoning.ProviderOpenAI)
		assert.Equal(t, reasoning.RejectsSamplingWhileThinking(model), support.RejectsSampling,
			"%s: the hint and the wire disagree about sampling", model)
	}
}
