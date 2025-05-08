package wikipedia

import (
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/internal/httprr"

	"github.com/stretchr/testify/require"
)

const _userAgent = "langchaingo test (https://github.com/vxcontrol/langchaingo)"

func TestWikipedia(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	// Setup httprr for HTTP requests
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	t.Cleanup(func() { rr.Close() })

	tool := New(_userAgent, WithHTTPClient(rr.Client()))
	_, err := tool.Call(ctx, "america")
	require.NoError(t, err)
}
