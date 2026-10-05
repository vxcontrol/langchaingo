package imageutil

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	pngSignature  = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n', 0, 0, 0, 0x0D}
	jpegSignature = []byte{0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10, 'J', 'F', 'I', 'F'}
)

func TestDownloadImageData(t *testing.T) {
	tests := []struct {
		name       string
		serverFunc func(w http.ResponseWriter, r *http.Request)
		wantType   string
		wantData   []byte
		wantErr    bool
		wantErrMsg string
	}{
		{
			name: "successful PNG download",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "image/png")
				w.Write([]byte{0x89, 0x50, 0x4E, 0x47}) // PNG header
			},
			wantType: "png",
			wantData: []byte{0x89, 0x50, 0x4E, 0x47},
			wantErr:  false,
		},
		{
			name: "successful JPEG download",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "image/jpeg")
				w.Write([]byte{0xFF, 0xD8, 0xFF, 0xE0}) // JPEG header
			},
			wantType: "jpeg",
			wantData: []byte{0xFF, 0xD8, 0xFF, 0xE0},
			wantErr:  false,
		},
		{
			name: "successful GIF download",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "image/gif")
				w.Write([]byte("GIF89a"))
			},
			wantType: "gif",
			wantData: []byte("GIF89a"),
			wantErr:  false,
		},
		{
			name: "content type without a slash and bytes that are no image",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "imagepng")
				w.Write([]byte{0x89, 0x50, 0x4E, 0x47})
			},
			wantErr:    true,
			wantErrMsg: `url does not point to an image: content type "imagepng"`,
		},
		{
			name: "content type with an extra part and bytes that are no image",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "image/png/extra")
				w.Write([]byte{0x89, 0x50, 0x4E, 0x47})
			},
			wantErr:    true,
			wantErrMsg: `url does not point to an image: content type "image/png/extra"`,
		},
		{
			name: "server error",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
			},
			wantErr:    true,
			wantErrMsg: "failed to fetch image from url: 500 Internal Server Error",
		},
		{
			name: "not found page",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/html")
				w.WriteHeader(http.StatusNotFound)
				w.Write([]byte("<html>missing</html>"))
			},
			wantErr:    true,
			wantErrMsg: "failed to fetch image from url: 404 Not Found",
		},
		{
			name: "no content type, PNG bytes",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header()["Content-Type"] = nil
				w.Write(pngSignature)
			},
			wantType: "png",
			wantData: pngSignature,
		},
		{
			name: "octet-stream content type, JPEG bytes",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/octet-stream")
				w.Write(jpegSignature)
			},
			wantType: "jpeg",
			wantData: jpegSignature,
		},
		{
			name: "content type with parameters",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "image/PNG; charset=binary")
				w.Write(pngSignature)
			},
			wantType: "png",
			wantData: pngSignature,
		},
		{
			name: "html page served as success",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/html; charset=utf-8")
				w.Write([]byte("<!DOCTYPE html><html>login</html>"))
			},
			wantErr:    true,
			wantErrMsg: `url does not point to an image: content type "text/html; charset=utf-8", content sniffed as "text/html"`,
		},
		{
			name: "empty response",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "image/png")
			},
			wantErr:    true,
			wantErrMsg: "url does not point to an image: empty body (200 OK)",
		},
		{
			name: "no content",
			serverFunc: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "image/png")
				w.WriteHeader(http.StatusNoContent)
			},
			wantErr:    true,
			wantErrMsg: "url does not point to an image: empty body (204 No Content)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(tt.serverFunc))
			defer server.Close()

			imageType, data, err := DownloadImageData(server.URL)

			if tt.wantErr {
				require.Error(t, err)
				if tt.wantErrMsg != "" {
					assert.Contains(t, err.Error(), tt.wantErrMsg)
				}
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.wantType, imageType)
				assert.Equal(t, tt.wantData, data)
			}
		})
	}
}

func TestDownloadImageData_InvalidURL(t *testing.T) {
	// Test with invalid URL
	_, _, err := DownloadImageData("http://[::1]:99999/invalid")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to fetch image from url")
}
