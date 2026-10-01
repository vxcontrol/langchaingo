package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func binaryContentServer(t *testing.T) (*LLM, *[]byte, *atomic.Int32) {
	t.Helper()

	var raw []byte
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"gpt-4.1",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"a cat"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithBaseURL(srv.URL), WithToken("token"), WithModel("gpt-4.1"))
	require.NoError(t, err)
	return llm, &raw, &calls
}

func TestABinaryImageGoesAsADataURL(t *testing.T) {
	t.Parallel()

	llm, raw, _ := binaryContentServer(t)

	_, err := llm.GenerateContent(t.Context(), []llms.MessageContent{{
		Role: llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{
			llms.TextContent{Text: "what is this?"},
			llms.BinaryContent{MIMEType: "image/png", Data: []byte{1, 2, 3}},
		},
	}})
	require.NoError(t, err)

	var sent struct {
		Messages []struct {
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(*raw, &sent))
	require.Len(t, sent.Messages, 1)
	require.Len(t, sent.Messages[0].Content, 2)
	assert.JSONEq(t, `{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}`,
		string(sent.Messages[0].Content[1]))
}

func TestABinaryPartThatIsNotAnImageIsRefusedBeforeTheRequest(t *testing.T) {
	t.Parallel()

	llm, _, calls := binaryContentServer(t)

	_, err := llm.GenerateContent(t.Context(), []llms.MessageContent{{
		Role: llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{
			llms.TextContent{Text: "summarize"},
			llms.BinaryContent{MIMEType: "application/pdf", Data: []byte("%PDF")},
		},
	}})

	require.ErrorIs(t, err, ErrUnsupportedContentType)
	assert.Contains(t, err.Error(), "application/pdf")
	assert.Zero(t, calls.Load())
}
