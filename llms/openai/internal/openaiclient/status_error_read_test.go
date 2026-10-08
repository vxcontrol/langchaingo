package openaiclient

import (
	"errors"
	"net/http"
	"testing"
	"testing/iotest"

	"github.com/stretchr/testify/require"
)

func TestABodyThatFailsToReadKeepsTheStatus(t *testing.T) {
	t.Parallel()

	err := statusError(http.StatusRequestEntityTooLarge, iotest.ErrReader(errors.New("connection reset")))

	var statusErr *StatusError
	require.ErrorAs(t, err, &statusErr)
	require.Equal(t, http.StatusRequestEntityTooLarge, statusErr.StatusCode)
	require.EqualError(t, err, "API returned unexpected status code: 413")
}
