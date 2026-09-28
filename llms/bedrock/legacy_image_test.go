package bedrock_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/bedrock/internal/bedrockclient"
)

const novaAnswer = `{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn",` +
	`"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`

func TestTheLegacyNovaDoorSendsAPictureAsAPicture(t *testing.T) {
	t.Parallel()

	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10}

	llm, sent := legacyLLMCapturing(t, novaAnswer, bedrock.WithModel("amazon.nova-lite-v1:0"))

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
			Content []struct {
				Image *struct {
					Format string `json:"format"`
					Source struct {
						Bytes string `json:"bytes"`
					} `json:"source"`
				} `json:"image"`
			} `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal([]byte(*sent), &payload))

	var found bool
	for _, m := range payload.Messages {
		for _, block := range m.Content {
			if block.Image == nil {
				continue
			}
			found = true
			assert.Equal(t, "jpeg", block.Image.Format,
				"the mime type decides the format the vendor is told")
			assert.Equal(t, base64.StdEncoding.EncodeToString(jpeg), block.Image.Source.Bytes,
				"the bytes travel unchanged")
		}
	}
	require.True(t, found, "the picture must reach the wire as an image block")
}

func TestTheLegacyNovaDoorRefusesAPictureItCannotName(t *testing.T) {
	t.Parallel()

	llm, sent := legacyLLMCapturing(t, novaAnswer, bedrock.WithModel("amazon.nova-lite-v1:0"))

	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.BinaryPart("image/heic", []byte{0x00, 0x01})},
	}})

	require.ErrorIs(t, err, bedrockclient.ErrUnsupportedImageFormat,
		"the legacy door must refuse by name what its Converse sibling refuses")
	assert.Empty(t, *sent, "an image the door cannot name must not reach the vendor at all")
}

func TestTheLegacyNovaDoorLeavesOutASystemPromptNobodyWrote(t *testing.T) {
	t.Parallel()

	llm, sent := legacyLLMCapturing(t, novaAnswer, bedrock.WithModel("amazon.nova-lite-v1:0"))

	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")})
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal([]byte(*sent), &payload))

	_, present := payload["system"]
	assert.False(t, present,
		"an absent system prompt leaves the key off the wire, as it does on the Converse door")
}

func TestTheLegacyNovaDoorSendsASystemPromptSomebodyWrote(t *testing.T) {
	t.Parallel()

	llm, sent := legacyLLMCapturing(t, novaAnswer, bedrock.WithModel("amazon.nova-lite-v1:0"))

	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "you are a hotel receptionist"),
		llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?"),
	})
	require.NoError(t, err)

	var payload struct {
		System []struct {
			Text string `json:"text"`
		} `json:"system"`
	}
	require.NoError(t, json.Unmarshal([]byte(*sent), &payload))
	require.Len(t, payload.System, 1)
	assert.Equal(t, "you are a hotel receptionist", payload.System[0].Text)
}
