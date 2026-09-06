package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestAnEncryptedThoughtArrivesAndTravelsBack(t *testing.T) {
	t.Parallel()

	const encrypted = "EqQBCgIYAhIM1gbcDa9GJwZA2b"

	t.Run("the answer survives a block the door cannot read", func(t *testing.T) {
		t.Parallel()

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_test","type":"message","role":"assistant",`+
				`"model":"claude-opus-4-6","content":[`+
				`{"type":"redacted_thinking","data":"`+encrypted+`"},`+
				`{"type":"text","text":"sixty rooms are free"}],`+
				`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(srv.Close)

		llm, err := anthropic.New(anthropic.WithToken("test-key"),
			anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-opus-4-6"))
		require.NoError(t, err)

		resp, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{{
				Role:  llms.ChatMessageTypeHuman,
				Parts: []llms.ContentPart{llms.TextPart("how many rooms are free?")},
			}}, llms.WithMaxTokens(64))
		require.NoError(t, err, "a block the door cannot read must not cost the caller the answer")
		require.Len(t, resp.Choices, 1)

		assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
		require.NotNil(t, resp.Choices[0].Reasoning, "the encrypted half is still reasoning")
		assert.Equal(t, encrypted, string(resp.Choices[0].Reasoning.Redacted))
		assert.False(t, resp.Choices[0].Reasoning.IsEmpty(),
			"a turn carrying an encrypted thought is not an empty one")
	})

	t.Run("the block goes back to the vendor unchanged", func(t *testing.T) {
		t.Parallel()

		var body []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_test","type":"message","role":"assistant",`+
				`"model":"claude-opus-4-6","content":[{"type":"text","text":"ok"}],`+
				`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(srv.Close)

		llm, err := anthropic.New(anthropic.WithToken("test-key"),
			anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-opus-4-6"))
		require.NoError(t, err)

		_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("hi")}},
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.TextPartWithReasoning("", &reasoning.ContentReasoning{Redacted: []byte(encrypted)}),
			}},
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("and now?")}},
		}, llms.WithMaxTokens(64))
		require.NoError(t, err)

		var payload struct {
			Messages []struct {
				Role    string           `json:"role"`
				Content []map[string]any `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.Unmarshal(body, &payload))

		var blocks []map[string]any
		for _, m := range payload.Messages {
			if m.Role == "assistant" {
				blocks = append(blocks, m.Content...)
			}
		}
		require.NotEmpty(t, blocks, "the assistant turn must reach the wire")
		assert.Contains(t, blocks, map[string]any{"type": "redacted_thinking", "data": encrypted},
			"the encrypted thought goes back exactly as it came")
	})
}
