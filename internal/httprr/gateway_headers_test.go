package httprr

import (
	"bytes"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type gatewayRoundTripper struct{}

func (gatewayRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header: http.Header{
			"Content-Type":        []string{"application/json"},
			"X-Litellm-Key-Spend": []string{"3783.008359056675"},
			"X-Litellm-Model-Id":  []string{"6915d127a6e7"},
		},
		Body:          io.NopCloser(strings.NewReader(`{"ok":true}`)),
		ContentLength: 11,
		Request:       req,
	}, nil
}

func repositoryRoot(t *testing.T) string {
	t.Helper()

	dir, err := os.Getwd()
	require.NoError(t, err)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		require.NotEqual(t, parent, dir, "go.mod not found above %s", dir)
		dir = parent
	}
}

func TestNoRecordingInThisRepositoryCarriesGatewayHeaders(t *testing.T) {
	root := repositoryRoot(t)
	scanned := 0

	require.NoError(t, filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() || filepath.Ext(path) != ".httprr" {
			return err //nolint:wrapcheck // the walk reports its own error unchanged
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err //nolint:wrapcheck // the walk reports its own error unchanged
		}
		scanned++
		name, _ := filepath.Rel(root, path)
		assert.NotContains(t, strings.ToLower(string(data)), "x-litellm-",
			"%s carries the gateway's headers: they hold the key's spend and the deployment id", name)
		return nil
	}))
	assert.Positive(t, scanned, "the sweep found no recordings at all, so it proves nothing")
}

func TestTheRecorderStripsGatewayHeadersFromAResponse(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gateway.httprr")
	defer setRecordForTesting(".*")()

	rr, err := Open(path, gatewayRoundTripper{})
	require.NoError(t, err)
	require.True(t, rr.Recording())

	client := &http.Client{Transport: rr}
	resp, err := client.Post("https://bedrock-runtime.us-east-1.amazonaws.com/model/x/invoke",
		"application/json", bytes.NewReader([]byte(`{}`)))
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	require.NoError(t, rr.Close())

	recorded, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Contains(t, string(recorded), "Content-Type", "the recording kept the ordinary headers")
	assert.NotContains(t, strings.ToLower(string(recorded)), "x-litellm-",
		"a recording made through the gateway must not keep its headers")
}
