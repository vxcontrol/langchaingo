package arxiv

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/internal/httprr"
)

func TestNew(t *testing.T) {
	t.Parallel()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	tool, err := New(2, DefaultUserAgent, WithHTTPClient(rr.Client()))
	require.NoError(t, err)

	call, err := tool.Call(t.Context(), "electron")
	require.NoError(t, err)
	require.Contains(t, call, "Title:")
}
