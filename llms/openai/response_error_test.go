package openai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func answerWith(t *testing.T, body string) (*llms.ContentResponse, error) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithBaseURL(srv.URL), WithToken("token"), WithModel("openai/gpt-4.1"))
	require.NoError(t, err)

	return llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
}

func TestAResponseFinishedWithAnErrorIsAProviderFailure(t *testing.T) {
	t.Parallel()

	resp, err := answerWith(t, `{"id":"gen-1","object":"chat.completion","created":1,"model":"openai/gpt-4.1",`+
		`"choices":[{"index":0,"message":{"role":"assistant","content":"partial output"},"finish_reason":"error",`+
		`"error":{"code":502,"message":"Provider disconnected mid-stream"}}]}`)

	require.ErrorIs(t, err, llms.ErrStreamFailed)
	assert.Contains(t, err.Error(), "Provider disconnected mid-stream")
	require.NotNil(t, resp)
	assert.Equal(t, "partial output", resp.Choices[0].Content)
}

func TestAnErrorBodyServedWithStatus200IsAProviderFailure(t *testing.T) {
	t.Parallel()

	_, err := answerWith(t, `{"error":{"code":502,"message":"Provider returned error"}}`)

	require.ErrorIs(t, err, llms.ErrStreamFailed)
	assert.Contains(t, err.Error(), "Provider returned error")
}
