package streamend

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

var errUserStop = errors.New("user pressed stop")

func TestAStreamThatSimplyEndedIsIncomplete(t *testing.T) {
	t.Parallel()

	err := Incomplete(t.Context(), nil)

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	assert.Equal(t, llms.ErrIncompleteStream.Error(), err.Error())
}

func TestTheReadErrorStaysInTheChain(t *testing.T) {
	t.Parallel()

	readErr := errors.New("connection reset by peer")

	err := Incomplete(t.Context(), readErr)

	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.ErrorIs(t, err, readErr)
	assert.Equal(t, llms.ErrIncompleteStream.Error()+": connection reset by peer", err.Error())
}

func TestAnEndedContextGivesBothItsErrorAndItsCause(t *testing.T) {
	t.Parallel()

	for name, end := range map[string]func(context.Context) context.Context{
		"cancel": func(parent context.Context) context.Context {
			ctx, cancel := context.WithCancel(parent)
			cancel()
			return ctx
		},
		"cancel with cause": func(parent context.Context) context.Context {
			ctx, cancel := context.WithCancelCause(parent)
			cancel(errUserStop)
			return ctx
		},
		"deadline with cause": func(parent context.Context) context.Context {
			ctx, cancel := context.WithTimeoutCause(parent, 0, errUserStop)
			t.Cleanup(cancel)
			return ctx
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			ctx := end(t.Context())

			err := Incomplete(ctx, nil)

			require.ErrorIs(t, err, llms.ErrIncompleteStream)
			require.ErrorIs(t, err, ctx.Err())
			require.ErrorIs(t, err, context.Cause(ctx))
		})
	}
}

func TestAReadErrorThatIsTheCauseIsNotRepeated(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(errUserStop)

	err := Incomplete(ctx, errUserStop)

	require.ErrorIs(t, err, context.Canceled)
	require.ErrorIs(t, err, errUserStop)
	assert.Equal(t, llms.ErrIncompleteStream.Error()+": user pressed stop: context canceled", err.Error())
}
