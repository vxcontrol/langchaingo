package imageutil

import (
	"net/http"
	"path/filepath"
	"testing"

	"github.com/vxcontrol/langchaingo/httputil"
	"github.com/vxcontrol/langchaingo/internal/httprr"

	"github.com/stretchr/testify/require"
)

// requireHttprrRecording replays the test's recording, or records it when
// -httprecord matches its file. The URLs are public, so a recording run needs
// no credentials; without one and without a recording the test skips.
func requireHttprrRecording(t *testing.T) *httprr.RecordReplay {
	t.Helper()

	recording, err := httprr.Recording(filepath.Join("testdata", httprr.CleanFileName(t.Name())+".httprr"))
	require.NoError(t, err)
	if !recording {
		httprr.SkipIfNoCredentialsAndRecordingMissing(t)
	}

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	return rr
}

func TestDownloadImageData_Integration(t *testing.T) {
	// Setup HTTP record/replay
	rr := requireHttprrRecording(t)
	defer rr.Close()

	// Replace httputil.DefaultClient with httprr client
	oldClient := httputil.DefaultClient
	httputil.DefaultClient = rr.Client()
	defer func() {
		httputil.DefaultClient = oldClient
	}()

	// Test downloading a PNG image
	imageType, data, err := DownloadImageData("https://placehold.co/150x150/FF0000/FFFFFF.png?text=Test")
	require.NoError(t, err)
	require.Equal(t, "png", imageType)
	require.NotEmpty(t, data)
}

func TestDownloadImageData_JPEG(t *testing.T) {
	// Setup HTTP record/replay
	rr := requireHttprrRecording(t)
	defer rr.Close()

	// Replace httputil.DefaultClient with httprr client
	oldClient := httputil.DefaultClient
	httputil.DefaultClient = rr.Client()
	defer func() {
		httputil.DefaultClient = oldClient
	}()

	// Test downloading a JPEG image
	imageType, data, err := DownloadImageData("https://placehold.co/150.jpg")
	require.NoError(t, err)
	require.Equal(t, "jpeg", imageType)
	require.NotEmpty(t, data)
}

func TestDownloadImageData_InvalidURL_Integration(t *testing.T) {
	// A URL without a scheme is rejected by the transport before any
	// connection is made, so this needs no recording.
	_, _, err := DownloadImageData("not-a-valid-url")
	require.ErrorContains(t, err, "unsupported protocol scheme")
}

func TestDownloadImageData_NotFound(t *testing.T) {
	// Setup HTTP record/replay
	rr := requireHttprrRecording(t)
	defer rr.Close()

	// Replace httputil.DefaultClient with httprr client
	oldClient := httputil.DefaultClient
	httputil.DefaultClient = rr.Client()
	defer func() {
		httputil.DefaultClient = oldClient
	}()

	_, _, err := DownloadImageData("https://httpbin.org/image/missing")
	require.ErrorContains(t, err, "failed to fetch image from url: 404 Not Found")
}

func TestDownloadImageData_InvalidMimeType(t *testing.T) {
	// Setup HTTP record/replay
	rr := requireHttprrRecording(t)
	defer rr.Close()

	// Replace httputil.DefaultClient with httprr client
	oldClient := httputil.DefaultClient
	httputil.DefaultClient = rr.Client()
	defer func() {
		httputil.DefaultClient = oldClient
	}()

	_, _, err := DownloadImageData("https://httpbin.org/robots.txt")
	require.ErrorContains(t, err, `url does not point to an image: content type "text/plain"`)
}
