package ollama

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolCallArgumentsKeepIntegersWiderThanFloat64(t *testing.T) {
	t.Parallel()

	o := &LLM{}
	converted, err := o.convertToolCall(llms.ToolCall{
		ID:   "1",
		Type: "function",
		FunctionCall: &llms.FunctionCall{
			Name:      "charge",
			Arguments: `{"account":9007199254740993,"cents":12345678901234567}`,
		},
	})
	require.NoError(t, err)

	account, ok := converted.Function.Arguments.Get("account")
	require.True(t, ok)
	assert.Equal(t, int64(9007199254740993), account)

	cents, ok := converted.Function.Arguments.Get("cents")
	require.True(t, ok)
	assert.Equal(t, int64(12345678901234567), cents)
}

func TestToolCallArgumentsKeepTheOrderTheModelSent(t *testing.T) {
	t.Parallel()

	o := &LLM{}
	converted, err := o.convertToolCall(llms.ToolCall{
		ID:           "1",
		Type:         "function",
		FunctionCall: &llms.FunctionCall{Name: "f", Arguments: `{"zebra":1,"alpha":2,"middle":3}`},
	})
	require.NoError(t, err)

	var keys []string
	for key := range converted.Function.Arguments.All() {
		keys = append(keys, key)
	}
	assert.Equal(t, []string{"zebra", "alpha", "middle"}, keys)
}
