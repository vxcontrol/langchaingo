package httprr

import (
	"bytes"
	"compress/gzip"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
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
			"Content-Type":                           []string{"application/json"},
			"X-Litellm-Key-Spend":                    []string{"3783.008359056675"},
			"X-Litellm-Model-Id":                     []string{"6915d127a6e7"},
			"Llm_provider-Anthropic-Organization-Id": []string{"1904456a-0000-0000-0000-000000000000"},
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

func readRecording(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil || !strings.HasSuffix(path, ".gz") {
		return data, err //nolint:wrapcheck // the walk reports its own error unchanged
	}
	zr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err //nolint:wrapcheck // the walk reports its own error unchanged
	}
	defer zr.Close()
	return io.ReadAll(zr)
}

var accountHeader = regexp.MustCompile(`(?im)^(openai-project|openai-organization): *([^\r\n]*)`)

func droppedHeaderLines() []string {
	lines := make([]string, 0, len(identifyingHeaders)+len(gatewayHeaderPrefixes))
	for _, name := range identifyingHeaders {
		lines = append(lines, "\n"+strings.ToLower(name)+":")
	}
	for _, prefix := range gatewayHeaderPrefixes {
		lines = append(lines, "\n"+prefix)
	}
	return lines
}

func TestNoRecordingInThisRepositoryCarriesGatewayHeaders(t *testing.T) {
	root := repositoryRoot(t)
	scanned := map[string]int{}

	require.NoError(t, filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err //nolint:wrapcheck // the walk reports its own error unchanged
		}
		kind := ""
		switch {
		case strings.HasSuffix(path, ".httprr"):
			kind = ".httprr"
		case strings.HasSuffix(path, ".httprr.gz"):
			kind = ".httprr.gz"
		default:
			return nil
		}
		data, err := readRecording(path)
		if err != nil {
			return err
		}
		scanned[kind]++
		name, _ := filepath.Rel(root, path)
		lowered := strings.ToLower(string(data))
		for _, line := range droppedHeaderLines() {
			assert.False(t, strings.Contains(lowered, line), "%s carries %q, which the recorder drops", name,
				strings.TrimSpace(line))
		}
		for _, m := range accountHeader.FindAllStringSubmatch(string(data), -1) {
			assert.Contains(t, []string{"proj_lcgo-tst", "lcgo-tst"}, strings.TrimSpace(m[2]),
				"%s carries a real %s", name, m[1])
		}
		return nil
	}))
	assert.Positive(t, scanned[".httprr"], "the sweep found no recordings at all, so it proves nothing")
	assert.Positive(t, scanned[".httprr.gz"], "the replayer also reads compressed recordings")
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
	assert.NotContains(t, strings.ToLower(string(recorded)), "llm_provider-",
		"the gateway forwards the vendor's headers under its own prefix")
}
