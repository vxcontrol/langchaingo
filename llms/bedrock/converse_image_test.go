package bedrock_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/bedrock/internal/bedrockclient"
)

func TestConverseSendsAPictureAsAPicture(t *testing.T) {
	t.Parallel()

	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10}

	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},`+
			`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv,
		bedrock.WithModel("amazon.nova-lite-v1:0"), bedrock.WithConverseAPI())

	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{{
		Role: llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{
			llms.TextPart("what is on this picture?"),
			llms.BinaryPart("image/jpeg", jpeg),
		},
	}})
	require.NoError(t, err)

	var payload struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal([]byte(body), &payload))

	var image map[string]any
	for _, m := range payload.Messages {
		for _, block := range m.Content {
			if got, ok := block["image"].(map[string]any); ok {
				image = got
			}
		}
	}
	require.NotNil(t, image, "the picture must reach the wire as an image block, not as text")
	assert.Equal(t, "jpeg", image["format"], "the mime type decides the format the vendor is told")

	source, ok := image["source"].(map[string]any)
	require.True(t, ok, "an image block carries its source")
	assert.Equal(t, base64.StdEncoding.EncodeToString(jpeg), source["bytes"],
		"the bytes travel unchanged, not as a text block full of replacement runes")
}

func TestConverseRefusesAPictureItCannotName(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},`+
			`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv,
		bedrock.WithModel("amazon.nova-lite-v1:0"), bedrock.WithConverseAPI())

	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.BinaryPart("image/heic", []byte{0x00, 0x01})},
	}})
	require.ErrorIs(t, err, bedrockclient.ErrUnsupportedImageFormat,
		"the door must refuse it by name; the SDK's own validation error would prove nothing about this guard")
}
