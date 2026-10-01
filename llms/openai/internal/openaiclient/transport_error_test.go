package openaiclient

import (
	"context"
	"errors"
	"net"
	"net/url"
	"os"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestASanitizedTransportErrorKeepsItsCauseForErrorsIs(t *testing.T) {
	t.Parallel()

	for _, cause := range []error{context.DeadlineExceeded, context.Canceled} {
		err := sanitizeHTTPError(&url.Error{Op: "Post", URL: "https://api.example.com/v1?key=secret", Err: cause})
		require.ErrorIs(t, err, cause)
		assert.NotContains(t, err.Error(), "secret")
	}
}

func TestASanitizedNetworkErrorNamesItsClass(t *testing.T) {
	t.Parallel()

	refused := &net.OpError{Op: "dial", Net: "tcp", Err: os.NewSyscallError("connect", syscall.ECONNREFUSED)}
	err := sanitizeHTTPError(&url.Error{Op: "Post", URL: "https://api.example.com/v1?key=secret", Err: refused})
	require.ErrorIs(t, err, syscall.ECONNREFUSED)
	assert.Equal(t, "network error: failed to reach API server: connection refused", err.Error())

	notFound := &net.DNSError{Err: "no such host", Name: "api.example.invalid", IsNotFound: true}
	err = sanitizeHTTPError(&url.Error{Op: "Post", URL: "https://api.example.invalid/v1?key=secret", Err: notFound})
	var dnsErr *net.DNSError
	require.True(t, errors.As(err, &dnsErr))
	assert.Equal(t, "network error: failed to reach API server: host not found", err.Error())
}
