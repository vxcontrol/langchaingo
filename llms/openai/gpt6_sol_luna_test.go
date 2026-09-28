package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vxcontrol/langchaingo/llms"
)

func TestSolAndLunaClampMaxTheVendorRefuses(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-6-sol", "gpt-6-luna"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			body, err := wireBodyOf(t, model, nil, llms.WithReasoning(llms.ReasoningMax, 0))
			require.NoError(t, err)
			assert.Equal(t, "xhigh", body["reasoning_effort"],
				"the card lists max, but the vendor answers 400 to it with this model and names none, low, medium, high and xhigh")
		})
	}
}
