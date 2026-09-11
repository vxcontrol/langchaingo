package huggingface

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnAdaptiveRequestWithNoEffortLeavesTheDepthToTheVendor(t *testing.T) {
	t.Parallel()

	body := bodyOfCall(t, oneMessage(), llms.WithAdaptiveReasoning(""))
	assert.NotContains(t, body, "reasoning_effort",
		"the caller handed the depth to the model; inventing high spends its tokens")
}

func TestAnAdaptiveRequestWithAnEffortStillCarriesIt(t *testing.T) {
	t.Parallel()

	body := bodyOfCall(t, oneMessage(), llms.WithAdaptiveReasoning(llms.ReasoningLow))
	require.Contains(t, body, "reasoning_effort")
	assert.Equal(t, "low", body["reasoning_effort"])
}
