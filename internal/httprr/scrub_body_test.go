package httprr

import (
	"bytes"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

type cannedTransport struct {
	header http.Header
	body   []byte
	length int64
}

func (c cannedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp := &http.Response{
		StatusCode:    http.StatusOK,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        c.header.Clone(),
		Body:          io.NopCloser(bytes.NewReader(c.body)),
		ContentLength: c.length,
		Request:       req,
	}
	if c.length < 0 {
		resp.TransferEncoding = []string{"chunked"}
	}
	return resp, nil
}

func TestARecordedBodyReplaysUnchanged(t *testing.T) {
	t.Parallel()

	body := bytes.Repeat([]byte("0123456789abcdef"), 100_000/16)
	for name, transport := range map[string]cannedTransport{
		"chunked": {header: http.Header{"Content-Type": {"application/x-ndjson"}}, body: body, length: -1},
		"head grows when scrubbed": {
			header: http.Header{"Content-Type": {"application/json"}, "Openai-Organization": {"o"}},
			body:   body, length: int64(len(body)),
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			file := t.TempDir() + "/rr"
			get := func(rr *RecordReplay) []byte {
				t.Helper()
				req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://example.invalid/", nil)
				require.NoError(t, err)
				resp, err := rr.RoundTrip(req)
				require.NoError(t, err)
				defer resp.Body.Close()
				got, err := io.ReadAll(resp.Body)
				require.NoError(t, err)
				return got
			}

			rr, err := create(file, transport)
			require.NoError(t, err)
			get(rr)
			require.NoError(t, rr.Close())

			rr, err = open(file, nil)
			require.NoError(t, err)
			defer rr.Close()
			require.True(t, bytes.Equal(body, get(rr)), "the replayed body differs from the recorded one")
		})
	}
}
