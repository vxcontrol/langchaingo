package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
)

func sendImageURL(t *testing.T, url string) (json.RawMessage, int32, error) {
	t.Helper()

	var body []byte
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_test","type":"message","role":"assistant",`+
			`"model":"claude-sonnet-4-5","content":[{"type":"text","text":"a cat"}],`+
			`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-sonnet-4-5"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.ImageURLContent{URL: url}, llms.TextContent{Text: "what is this?"}},
	}}, llms.WithMaxTokens(64))
	if err != nil {
		return nil, calls.Load(), err
	}

	var sent struct {
		Messages []struct {
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &sent))
	require.Len(t, sent.Messages, 1)
	require.Len(t, sent.Messages[0].Content, 2)
	return sent.Messages[0].Content[0], calls.Load(), nil
}

func TestAnImageURLGoesAsAURLSource(t *testing.T) {
	t.Parallel()

	image, _, err := sendImageURL(t, "https://example.com/cat.png")

	require.NoError(t, err)
	assert.JSONEq(t, `{"type":"image","source":{"type":"url","url":"https://example.com/cat.png"}}`, string(image))
}

func TestADataURLImageGoesAsABase64Source(t *testing.T) {
	t.Parallel()

	image, _, err := sendImageURL(t, "data:image/png;base64,AQID")

	require.NoError(t, err)
	assert.JSONEq(t, `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}}`,
		string(image))
}

func TestAnImageURLOfAnotherSchemeIsRefusedBeforeTheRequest(t *testing.T) {
	t.Parallel()

	_, calls, err := sendImageURL(t, "file:///tmp/cat.png")

	require.ErrorIs(t, err, anthropic.ErrUnsupportedContentType)
	assert.Zero(t, calls)
}
