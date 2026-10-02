package imageutil

import (
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"

	"github.com/vxcontrol/langchaingo/httputil"
)

func DownloadImageData(url string) (string, []byte, error) {
	resp, err := httputil.DefaultClient.Get(url)
	if err != nil {
		return "", nil, fmt.Errorf("failed to fetch image from url: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return "", nil, fmt.Errorf("failed to fetch image from url: %s", resp.Status)
	}

	urlData, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, fmt.Errorf("failed to read image bytes: %w", err)
	}

	header := resp.Header.Get("Content-Type")
	mediaType, _, err := mime.ParseMediaType(header)
	if err != nil || !strings.HasPrefix(mediaType, "image/") {
		mediaType, _, _ = mime.ParseMediaType(http.DetectContentType(urlData))
	}
	subtype, isImage := strings.CutPrefix(mediaType, "image/")
	if !isImage || subtype == "" {
		return "", nil, fmt.Errorf("url does not point to an image: content type %q, content sniffed as %q",
			header, mediaType)
	}

	return subtype, urlData, nil
}
