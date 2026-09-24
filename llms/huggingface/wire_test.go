package huggingface

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type capturingTransport struct {
	body     []byte
	requests []*url.URL
	reply    string
}

func (c *capturingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	c.requests = append(c.requests, r.URL)
	if r.Body != nil {
		c.body, _ = io.ReadAll(r.Body)
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(c.reply)),
		Request:    r,
	}, nil
}

func callThroughTransport(t *testing.T, tr *capturingTransport, opts ...Option) {
	t.Helper()

	tr.reply = `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`
	llm, err := New(append([]Option{
		WithToken("t"),
		WithModel("gpt2"),
		WithHTTPClient(&http.Client{Transport: tr}),
	}, opts...)...)
	require.NoError(t, err)

	_, err = llm.Call(context.Background(), "hi")
	require.NoError(t, err)
}

func TestWithoutAnExplicitURLTheDoorAddressesTheRouter(t *testing.T) {
	t.Parallel()

	tr := &capturingTransport{}
	callThroughTransport(t, tr)

	require.Len(t, tr.requests, 1)
	assert.Equal(t, "router.huggingface.co", tr.requests[0].Host)
}

func TestAnExplicitURLSurvivesTheChoiceOfProvider(t *testing.T) {
	t.Parallel()

	tr := &capturingTransport{}
	tr.reply = `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`
	llm, err := New(
		WithToken("t"),
		WithModel("gpt2"),
		WithURL("https://proxy.example.test"),
		WithInferenceProvider("nebius"),
		WithHTTPClient(&http.Client{Transport: tr}),
	)
	require.NoError(t, err)

	_, err = llm.Call(context.Background(), "hi")
	require.NoError(t, err)

	require.Len(t, tr.requests, 1)
	assert.Equal(t, "proxy.example.test", tr.requests[0].Host)
}
