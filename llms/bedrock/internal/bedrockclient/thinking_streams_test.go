package bedrockclient

import (
	"context"
	"errors"
	"testing"

	"github.com/vxcontrol/langchaingo/llms/streaming"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func pendingToolCalls() map[int32]*converseToolCallBuilder {
	return map[int32]*converseToolCallBuilder{
		1: {id: "call_2", name: "second"},
		0: {id: "call_1", name: "first"},
	}
}

func TestASalvagedToolCallReachesTheConsumerToo(t *testing.T) {
	t.Parallel()

	var delivered []string
	calls, err := salvageToolCalls(t.Context(),
		func(_ context.Context, chunk streaming.Chunk) error {
			if chunk.Type == streaming.ChunkTypeToolCall {
				delivered = append(delivered, chunk.ToolCall.Name)
			}
			return nil
		}, pendingToolCalls())

	require.NoError(t, err)
	require.Len(t, calls, 2, "the final answer keeps both salvaged calls")
	assert.Equal(t, []string{"first", "second"}, delivered,
		"an agent living on chunks must see the calls the answer carries, in block order")
}

func TestAConsumerThatRefusesASalvagedCallStopsTheSalvage(t *testing.T) {
	t.Parallel()

	gaveUp := errors.New("the consumer gave up")
	calls, err := salvageToolCalls(t.Context(),
		func(context.Context, streaming.Chunk) error { return gaveUp }, pendingToolCalls())

	require.ErrorIs(t, err, gaveUp)
	assert.Empty(t, calls)
}

func TestASalvageWithoutAConsumerStillFillsTheAnswer(t *testing.T) {
	t.Parallel()

	calls, err := salvageToolCalls(t.Context(), nil, pendingToolCalls())
	require.NoError(t, err)
	assert.Len(t, calls, 2)
}
