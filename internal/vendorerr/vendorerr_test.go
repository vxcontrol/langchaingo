package vendorerr

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAPositionTheDoorCannotMapIsMinusOne(t *testing.T) {
	t.Parallel()

	for _, message := range []string{
		"Invalid `signature` in `thinking` block",
		"messages.7.content.0: Invalid `signature` in `thinking` block",
	} {
		var rejected *llms.ErrHistoryRejected
		require.ErrorAs(t, Classify(errors.New(message), 400, message, []int{1, 3}), &rejected, message)
		require.Equal(t, -1, rejected.Message, message)
	}
}
