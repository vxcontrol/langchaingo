package vendorerr

import (
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAPositionTheDoorCannotMapIsMinusOne(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		message string
		origins []int
	}{
		{"Invalid `signature` in `thinking` block", []int{1, 3}},
		{"messages.7.content.0: Invalid `signature` in `thinking` block", []int{1, 3}},
		{"messages.2.content.0: Invalid `signature` in `thinking` block", []int{1, 3}},
		{"messages.0.content.0: Invalid `signature` in `thinking` block", nil},
	} {
		var rejected *llms.ErrHistoryRejected
		require.ErrorAs(t, Classify(errors.New(tc.message), http.StatusBadRequest, tc.message, tc.origins), &rejected, tc.message)
		require.Equal(t, -1, rejected.Message, tc.message)
	}
}

func TestOnlyTheTooLargeStatusIsAnOverflow(t *testing.T) {
	t.Parallel()

	for _, status := range []int{
		http.StatusPreconditionFailed, http.StatusRequestURITooLong, http.StatusTooManyRequests,
		http.StatusInternalServerError, 529,
	} {
		err := errors.New("upstream error")
		require.Same(t, err, Classify(err, status, "upstream error", nil), "status %d", status)
	}
}
