package googleai

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolCallArgumentsKeepIntegersWiderThanFloat64(t *testing.T) {
	t.Parallel()

	parts, err := convertParts([]llms.ContentPart{llms.ToolCall{
		ID:   "call_1",
		Type: "function",
		FunctionCall: &llms.FunctionCall{
			Name:      "charge",
			Arguments: `{"account":9007199254740993,"cents":12345678901234567}`,
		},
	}})
	require.NoError(t, err)
	require.Len(t, parts, 1)
	require.NotNil(t, parts[0].FunctionCall)

	args := parts[0].FunctionCall.Args
	assert.Equal(t, int64(9007199254740993), args["account"])
	assert.Equal(t, int64(12345678901234567), args["cents"])
}
