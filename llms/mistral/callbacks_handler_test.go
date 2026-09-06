package mistral

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/callbacks"
)

type countingHandler struct{ callbacks.SimpleHandler }

func TestTheCallersCallbacksHandlerReachesTheModel(t *testing.T) {
	t.Parallel()

	handler := &countingHandler{}
	llm, err := New(WithAPIKey("test"), WithCallbacksHandler(handler))
	require.NoError(t, err)

	assert.Same(t, handler, llm.CallbacksHandler,
		"the handler the caller passed must be the one the model calls")
}

func TestWithoutACallbacksHandlerTheModelKeepsTheEmptyOne(t *testing.T) {
	t.Parallel()

	llm, err := New(WithAPIKey("test"))
	require.NoError(t, err)

	assert.Equal(t, callbacks.SimpleHandler{}, llm.CallbacksHandler,
		"the default stays the do-nothing handler")
}
