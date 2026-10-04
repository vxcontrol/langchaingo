package main

import (
	"bytes"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func gzipped(t *testing.T, data []byte) []byte {
	t.Helper()

	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	_, err := zw.Write(data)
	require.NoError(t, err)
	require.NoError(t, zw.Close())
	return buf.Bytes()
}

func gunzipFile(t *testing.T, path string) []byte {
	t.Helper()

	f, err := os.Open(path)
	require.NoError(t, err)
	defer f.Close()
	zr, err := gzip.NewReader(f)
	require.NoError(t, err)
	data, err := io.ReadAll(zr)
	require.NoError(t, err)
	return data
}

func TestFixKeepsTheRecordingTheReplayerWouldRead(t *testing.T) {
	older := time.Now().Add(-time.Hour)
	newer := time.Now()

	for _, tc := range []struct {
		name               string
		plainAt, gzipAt    time.Time
		wantPlainRecording bool
		wantHint           string
	}{
		{"there is no gzip", newer, time.Time{}, true, "gzip "},
		{"the plain recording is newer", newer, older, true, "gzip -f "},
		{"the gzip is newer", older, newer, false, "rm "},
		{"both carry the same time", newer, newer, false, "rm "},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Chdir(t.TempDir())

			plain := filepath.Join("llms", "acme", "testdata", "TestCall.httprr")
			plainRecording := []byte("httprr trace v1\n12 34\nGET / HTTP/1.1\r\n\r\nHTTP/1.1 200 OK\r\n\r\n")
			gzipRecording := []byte("httprr trace v1\n12 34\nGET /gz HTTP/1.1\r\n\r\nHTTP/1.1 200 OK\r\n\r\n")
			require.NoError(t, os.MkdirAll(filepath.Dir(plain), 0o755))
			require.NoError(t, os.WriteFile(plain, plainRecording, 0o644))
			require.NoError(t, os.Chtimes(plain, tc.plainAt, tc.plainAt))
			if !tc.gzipAt.IsZero() {
				require.NoError(t, os.WriteFile(plain+".gz", gzipped(t, gzipRecording), 0o644))
				require.NoError(t, os.Chtimes(plain+".gz", tc.gzipAt, tc.gzipAt))
			}

			require.ErrorContains(t, checkHttprrCompression(false), plain+": "+tc.wantHint+plain)
			require.FileExists(t, plain, "a check without -fix changes nothing")

			require.NoError(t, checkHttprrCompression(true))
			require.NoFileExists(t, plain)
			want := gzipRecording
			if tc.wantPlainRecording {
				want = plainRecording
			}
			require.Equal(t, string(want), string(gunzipFile(t, plain+".gz")))
			require.NoError(t, checkHttprrCompression(false))
		})
	}
}
