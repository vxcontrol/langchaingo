package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
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
		assert.Equal(t, []reasoning.Block{{Redacted: []byte(encrypted)}}, resp.Choices[0].Reasoning.Sequence())
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
				llms.TextPartWithReasoning("", reasoning.FromBlocks([]reasoning.Block{{Redacted: []byte(encrypted)}})),
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
		assert.Equal(t, []map[string]any{{"type": "redacted_thinking", "data": encrypted}}, blocks,
			"the encrypted thought goes back exactly as it came, and nothing else rides with it")
	})
}

func redactedThinkingStream(t *testing.T, encrypted string) *httptest.Server {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":"+
			"{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-6\","+
			"\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
		_, _ = io.WriteString(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\","+
			"\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\""+encrypted+"\"}}\n\n")
		_, _ = io.WriteString(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
		_, _ = io.WriteString(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\","+
			"\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		_, _ = io.WriteString(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\","+
			"\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"sixty rooms are free\"}}\n\n")
		_, _ = io.WriteString(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n")
		_, _ = io.WriteString(w, "event: message_delta\ndata: {\"type\":\"message_delta\","+
			"\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":9}}\n\n")
		_, _ = io.WriteString(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestAStreamedEncryptedThoughtDoesNotCostTheAnswer(t *testing.T) {
	t.Parallel()

	const encrypted = "EqQBCgIYAhIM1gbcDa9GJwZA2b"

	srv := redactedThinkingStream(t, encrypted)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-opus-4-6"))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart("how many rooms are free?")},
		}},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	require.NoError(t, err, "a streamed block the door cannot read must not cost the caller the answer")
	require.Len(t, resp.Choices, 1)

	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
	require.NotNil(t, resp.Choices[0].Reasoning, "the encrypted half is still reasoning on the streamed leg")
	assert.Equal(t, []reasoning.Block{{Redacted: []byte(encrypted)}}, resp.Choices[0].Reasoning.Sequence())
	assert.False(t, resp.Choices[0].Reasoning.IsEmpty())
}

func TestTwoEncryptedBlocksGoBackAsTwo(t *testing.T) {
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
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextContent{
			Text:      "answer",
			Reasoning: reasoning.FromBlocks([]reasoning.Block{{Redacted: []byte("first")}, {Redacted: []byte("second")}}),
		}}},
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("and now?")}},
	}, llms.WithMaxTokens(64))
	require.NoError(t, err)

	assert.Equal(t, 2, strings.Count(string(body), `"type":"redacted_thinking"`),
		"two encrypted blocks must reach the vendor as two")
}
