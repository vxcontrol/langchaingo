package arxiv

import (
	"io"
	"net/http"
	"strings"
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

type cannedFeed struct{ requests int }

func (c *cannedFeed) RoundTrip(req *http.Request) (*http.Response, error) {
	c.requests++
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"application/atom+xml"}},
		Body: io.NopCloser(strings.NewReader(`<feed xmlns="http://www.w3.org/2005/Atom"><entry>` +
			`<title>Canned paper</title><summary>s</summary><author><name>A</name></author></entry></feed>`)),
		Request: req,
	}, nil
}

func TestTheSearchGoesThroughTheGivenClient(t *testing.T) {
	t.Parallel()

	transport := &cannedFeed{}
	tool, err := New(1, DefaultUserAgent, WithHTTPClient(&http.Client{Transport: transport}))
	require.NoError(t, err)

	call, err := tool.Call(t.Context(), "electron")
	require.NoError(t, err)
	require.Contains(t, call, "Title: Canned paper")
	require.Equal(t, 1, transport.requests)
}
