package bedrock_test

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
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

type wireTurn struct {
	Role    string           `json:"role"`
	Content []map[string]any `json:"content"`
}

func converseTurns(t *testing.T, messages []llms.MessageContent) []wireTurn {
	t.Helper()

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
	_, err := llm.GenerateContent(context.Background(), messages)
	require.NoError(t, err)

	var payload struct {
		Messages []wireTurn `json:"messages"`
	}
	require.NoError(t, json.Unmarshal([]byte(body), &payload))
	return payload.Messages
}

func TestOneHumanTurnReachesConverseAsOneMessage(t *testing.T) {
	t.Parallel()

	turns := converseTurns(t, []llms.MessageContent{{
		Role: llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{
			llms.TextPart("what is on this picture?"),
			llms.BinaryPart("image/jpeg", []byte{0xFF, 0xD8}),
			llms.TextPart("and how many rooms are free?"),
		},
	}})

	require.Len(t, turns, 1, "one turn the caller wrote is one message on the wire, as on the legacy door")
	assert.Equal(t, "user", turns[0].Role)
	require.Len(t, turns[0].Content, 3, "every part of that turn travels, in the order it was written")
	assert.Contains(t, turns[0].Content[0], "text")
	assert.Contains(t, turns[0].Content[1], "image")
	assert.Contains(t, turns[0].Content[2], "text")
}

func TestAnAssistantTurnKeepsEveryTextItCarried(t *testing.T) {
	t.Parallel()

	turns := converseTurns(t, []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextPart("first I checked the ledger"),
			llms.TextPart("sixty rooms are free"),
		}},
		llms.TextParts(llms.ChatMessageTypeHuman, "thanks"),
	})

	require.Len(t, turns, 3)
	assert.Equal(t, "assistant", turns[1].Role)
	require.Len(t, turns[1].Content, 2, "a second text in the same turn is not dropped")
	assert.Equal(t, "first I checked the ledger", turns[1].Content[0]["text"])
	assert.Equal(t, "sixty rooms are free", turns[1].Content[1]["text"])
}
