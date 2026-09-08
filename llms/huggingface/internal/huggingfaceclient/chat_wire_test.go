package huggingfaceclient

import (
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func chatRequestOf(t *testing.T, opts []Option) (string, map[string]any) {
	t.Helper()

	var (
		path string
		raw  []byte
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)

	client, err := New("t", "gpt2", srv.URL, opts...)
	require.NoError(t, err)

	resp, err := client.RunInference(t.Context(), &InferenceRequest{
		Model:       "gpt2",
		Prompt:      "hi",
		Temperature: 0.5,
		MaxLength:   20,
	})
	require.NoError(t, err)
	assert.Equal(t, "stop", resp.StopReason)

	var body map[string]any
	require.NoError(t, json.Unmarshal(raw, &body))

	return path, body
}

func TestTheChatRequestAddressesTheDocumentedEndpoint(t *testing.T) {
	t.Parallel()

	path, body := chatRequestOf(t, nil)

	assert.Equal(t, "/v1/chat/completions", path)
	assert.Equal(t, []string{"max_tokens", "messages", "model", "stream", "temperature"},
		slices.Sorted(maps.Keys(body)))
}

func TestANamedProviderPrefixesTheChatEndpoint(t *testing.T) {
	t.Parallel()

	path, _ := chatRequestOf(t, []Option{WithProvider("nebius")})

	assert.Equal(t, "/nebius/v1/chat/completions", path)
}
