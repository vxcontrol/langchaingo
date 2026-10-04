package main

import (
	"bytes"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"testing"

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

func TestFixReplacesAnUncompressedRecordingWithItsGzip(t *testing.T) {
	t.Chdir(t.TempDir())

	plain := filepath.Join("llms", "acme", "testdata", "TestCall.httprr")
	recording := []byte("httprr trace v1\n12 34\nGET / HTTP/1.1\r\n\r\nHTTP/1.1 200 OK\r\n\r\n")
	require.NoError(t, os.MkdirAll(filepath.Dir(plain), 0o755))
	require.NoError(t, os.WriteFile(plain, recording, 0o644))
	require.NoError(t, os.WriteFile(plain+".gz", gzipped(t, []byte("an older recording")), 0o644))

	require.ErrorContains(t, checkHttprrCompression(false), plain)
	require.FileExists(t, plain, "a check without -fix changes nothing")

	require.NoError(t, checkHttprrCompression(true))
	require.NoFileExists(t, plain, "the replayer reads a plain recording before its gzip")
	require.Equal(t, recording, gunzipFile(t, plain+".gz"), "the plain recording is the newer one")
	require.NoError(t, checkHttprrCompression(false))
}
