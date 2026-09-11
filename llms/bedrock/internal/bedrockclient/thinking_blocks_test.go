package bedrockclient

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTwoEncryptedBlocksTravelBackAsTwo(t *testing.T) {
	t.Parallel()

	messages := []Message{
		{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"},
		{
			Role: llms.ChatMessageTypeAI, Content: "answer", Type: "text",
			Reasoning: &reasoning.ContentReasoning{Redacted: [][]byte{{0x01}, {0x02}}},
		},
	}

	contents, _, err := processInputMessagesAnthropic(messages)
	require.NoError(t, err)

	blocks := 0
	for _, message := range contents {
		for _, block := range message.Content {
			if block.Type == "redacted_thinking" {
				blocks++
			}
		}
	}
	assert.Equal(t, 2, blocks, "merging them would leave nothing to split on the way back")
}

func TestASystemMessageDoesNotSplitTwoHumanTurns(t *testing.T) {
	t.Parallel()

	client := NewConverseClient(nil)
	built, err := client.buildConverseInput(&ConverseInput{
		ModelID: "us.anthropic.claude-haiku-4-5-v1:0",
		Messages: []Message{
			{Role: llms.ChatMessageTypeHuman, Content: "first", Type: "text"},
			{Role: llms.ChatMessageTypeSystem, Content: "be brief", Type: "text"},
			{Role: llms.ChatMessageTypeHuman, Content: "second", Type: "text"},
		},
	})
	require.NoError(t, err)

	require.Len(t, built.Messages, 1, "the vendor refuses two user turns in a row")
	assert.Len(t, built.Messages[0].Content, 2, "both human blocks stay in that one turn")
}

func TestALegacyImageFormatTheVendorRefusesIsCaughtBeforeTheWire(t *testing.T) {
	t.Parallel()

	_, _, err := processInputMessagesAnthropic([]Message{{
		Role: llms.ChatMessageTypeHuman, Type: AnthropicMessageTypeImage,
		MimeType: "image/heic", Content: "x",
	}})
	require.ErrorIs(t, err, ErrUnsupportedImageFormat)
}

func TestALegacyImageFormatTheVendorTakesStillTravels(t *testing.T) {
	t.Parallel()

	contents, _, err := processInputMessagesAnthropic([]Message{{
		Role: llms.ChatMessageTypeHuman, Type: AnthropicMessageTypeImage,
		MimeType: "image/png", Content: "x",
	}})
	require.NoError(t, err)
	require.NotEmpty(t, contents)
}
