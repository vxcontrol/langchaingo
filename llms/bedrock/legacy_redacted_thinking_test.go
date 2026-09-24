package bedrock_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestTheLegacyPathCarriesTheEncryptedThought(t *testing.T) {
	t.Parallel()

	const (
		encrypted = "EqQBCgIYAhIM1gbcDa9GJwZA2b"
		model     = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	)

	t.Run("it arrives with the answer", func(t *testing.T) {
		t.Parallel()

		llm, _ := legacyLLMCapturing(t, `{"id":"x","type":"message","role":"assistant","model":"m",`+
			`"content":[{"type":"redacted_thinking","data":"`+encrypted+`"},`+
			`{"type":"text","text":"sixty rooms are free"}],"stop_reason":"end_turn",`+
			`"usage":{"input_tokens":1,"output_tokens":1}}`, bedrock.WithModel(model))

		resp, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")})
		require.NoError(t, err)
		require.Len(t, resp.Choices, 1)

		assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
		require.NotNil(t, resp.Choices[0].Reasoning,
			"a turn carrying an encrypted thought is not a turn without thought")
		assert.Equal(t, []reasoning.Block{{Redacted: []byte(encrypted)}}, resp.Choices[0].Reasoning.Sequence())
	})

	t.Run("it goes back to the vendor unchanged", func(t *testing.T) {
		t.Parallel()

		llm, sent := legacyLLMCapturing(t, `{"id":"x","type":"message","role":"assistant","model":"m",`+
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",`+
			`"usage":{"input_tokens":1,"output_tokens":1}}`, bedrock.WithModel(model))

		_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("hi")}},
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.TextPartWithReasoning("", reasoning.FromBlocks([]reasoning.Block{{Redacted: []byte(encrypted)}})),
			}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("and now?")}},
		})
		require.NoError(t, err)

		var payload struct {
			Messages []struct {
				Role    string           `json:"role"`
				Content []map[string]any `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.Unmarshal([]byte(*sent), &payload))

		var blocks []map[string]any
		for _, m := range payload.Messages {
			if m.Role == "assistant" {
				blocks = append(blocks, m.Content...)
			}
		}
		require.NotEmpty(t, blocks, "the assistant turn must reach the wire")
		assert.Contains(t, blocks, map[string]any{"type": "redacted_thinking", "data": encrypted},
			"the encrypted thought goes back exactly as it came, as the Converse path already does")
	})
}
