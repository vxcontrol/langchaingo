package bedrock_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
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

func turnShapes(turns []wireTurn) []string {
	shapes := make([]string, 0, len(turns))
	for _, turn := range turns {
		blocks := make([]string, 0, len(turn.Content))
		for _, block := range turn.Content {
			switch {
			case block["text"] != nil:
				blocks = append(blocks, fmt.Sprintf("text %v", block["text"]))
			case block["cachePoint"] != nil:
				blocks = append(blocks, "cache")
			case block["toolUse"] != nil:
				use, _ := block["toolUse"].(map[string]any)
				blocks = append(blocks, fmt.Sprintf("tool %v", use["toolUseId"]))
			case block["reasoningContent"] != nil:
				thought, _ := block["reasoningContent"].(map[string]any)
				text, _ := thought["reasoningText"].(map[string]any)
				blocks = append(blocks, fmt.Sprintf("thought %v/%v", text["text"], text["signature"]))
			default:
				blocks = append(blocks, fmt.Sprintf("%v", block))
			}
		}
		shapes = append(shapes, turn.Role+": "+strings.Join(blocks, ", "))
	}
	return shapes
}

func TestAnEmptyAssistantTurnNeverReachesConverse(t *testing.T) {
	t.Parallel()

	first := llms.TextParts(llms.ChatMessageTypeHuman, "first")
	second := llms.TextParts(llms.ChatMessageTypeHuman, "second")
	thought := reasoning.FromBlocks([]reasoning.Block{{Text: "plan", Signature: []byte("s1")}})

	for name, tc := range map[string]struct {
		history []llms.MessageContent
		want    []string
	}{
		"an empty text": {
			history: []llms.MessageContent{first, {Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextContent{}}}, second},
			want:    []string{"user: text first, text second"},
		},
		"a turn with no parts": {
			history: []llms.MessageContent{first, {Role: llms.ChatMessageTypeAI}, second},
			want:    []string{"user: text first, text second"},
		},
		"an empty text with a cache marker": {
			history: []llms.MessageContent{
				first,
				{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
					bedrock.WithCacheControl(llms.TextContent{}, bedrock.EphemeralCache()),
				}},
				second,
				llms.TextParts(llms.ChatMessageTypeAI, "answer"),
				llms.TextParts(llms.ChatMessageTypeHuman, "third"),
			},
			want: []string{"user: text first, text second", "assistant: text answer", "user: text third, cache"},
		},
		"thinking alone": {
			history: []llms.MessageContent{first, {Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.TextPartWithReasoning("", thought),
			}}, second},
			want: []string{"user: text first", "assistant: thought plan/s1", "user: text second"},
		},
		"a tool call alone": {
			history: []llms.MessageContent{first, {Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.ToolCall{ID: "A", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: `{"q":"A"}`}},
			}}, second},
			want: []string{"user: text first", "assistant: tool A", "user: text second"},
		},
		"an empty user text with a cache marker": {
			history: []llms.MessageContent{
				{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{
					llms.TextPart("first"),
					bedrock.WithCacheControl(llms.TextContent{}, bedrock.EphemeralCache()),
				}},
				llms.TextParts(llms.ChatMessageTypeAI, "answer"),
				second,
			},
			want: []string{"user: text first, cache", "assistant: text answer", "user: text second, cache"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tc.want, turnShapes(converseTurns(t, tc.history)))
		})
	}
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
