package anthropicclient

import (
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type doerFunc func(*http.Request) (*http.Response, error)

func (f doerFunc) Do(req *http.Request) (*http.Response, error) { return f(req) }

func TestAClientWithoutABaseURLSendsConcurrentCallsToTheDefault(t *testing.T) {
	t.Parallel()

	var mu sync.Mutex
	var urls []string
	client, err := New("test-api-key", "claude-sonnet-5", "", WithHTTPClient(doerFunc(func(req *http.Request) (*http.Response, error) {
		mu.Lock()
		urls = append(urls, req.URL.String())
		mu.Unlock()
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(`{"id":"m","type":"message","role":"assistant",` +
				`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)),
		}, nil
	})))
	require.NoError(t, err)

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			_, err := client.CreateMessage(t.Context(), &MessageRequest{
				Model:     "claude-sonnet-5",
				Messages:  []ChatMessage{{Role: "user", Content: []Content{TextContent{Type: "text", Text: "hi"}}}},
				MaxTokens: getIntPointer(16),
			})
			assert.NoError(t, err)
		})
	}
	wg.Wait()

	require.Len(t, urls, 8)
	for _, url := range urls {
		assert.True(t, strings.HasPrefix(url, DefaultBaseURL+"/"), url)
	}
}
