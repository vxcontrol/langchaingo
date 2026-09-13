package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func mistralAnswering(t *testing.T, message string) *LLM {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"mistral-small-latest",`+
			`"choices":[{"index":0,"message":`+message+`,"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(server.Close)

	llm, err := New(WithBaseURL(server.URL), WithToken("token"), WithModel("mistral-small-latest"))
	require.NoError(t, err)
	return llm
}

func TestAMistralAnswerWithThinkingGivesTheTextAndTheReasoning(t *testing.T) {
	t.Parallel()

	llm := mistralAnswering(t, `{"role":"assistant","content":[`+
		`{"type":"thinking","thinking":[{"type":"text","text":"17 times 23"},{"type":"text","text":" is 391."}],"closed":true},`+
		`{"type":"text","text":"391"}]}`)

	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "What is 17*23?"),
	}, llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)

	choice := resp.Choices[0]
	assert.Equal(t, "391", choice.Content)
	require.NotNil(t, choice.Reasoning)
	assert.Equal(t, "17 times 23 is 391.", choice.Reasoning.Content)
}

func TestAMistralToolCallWithThinkingKeepsTheCallAndTheReasoning(t *testing.T) {
	t.Parallel()

	llm := mistralAnswering(t, `{"role":"assistant","content":[`+
		`{"type":"thinking","thinking":[{"type":"text","text":"The tool knows."}],"closed":true}],`+
		`"tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{\"city\": \"Paris\"}"},"index":0}]}`)

	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "Population of Paris?"),
	}, llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)

	choice := resp.Choices[0]
	assert.Empty(t, choice.Content)
	require.Len(t, choice.ToolCalls, 1)
	assert.Equal(t, "lookup", choice.ToolCalls[0].FunctionCall.Name)
	require.NotNil(t, choice.Reasoning)
	assert.Equal(t, "The tool knows.", choice.Reasoning.Content)
}

func TestTheMistralStreamHandsOverTheEndOfThinkingAndTheStartOfTheAnswer(t *testing.T) {
	t.Parallel()

	deltas := []string{
		`{"role":"assistant","content":""}`,
		`{"content":[{"type":"thinking","thinking":[{"type":"text","text":"17 times"}]}]}`,
		`{"content":[{"type":"thinking","thinking":[{"type":"text","text":" 23"}],"closed":true}]}`,
		`{"content":[{"type":"thinking","thinking":[{"type":"text","text":"."}],"closed":true},{"type":"text","text":"3"}]}`,
		`{"content":"91"}`,
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i, delta := range deltas {
			finish := "null"
			if i == len(deltas)-1 {
				finish = `"stop"`
			}
			_, _ = io.WriteString(w, `data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"mistral-small-latest",`+
				`"choices":[{"index":0,"delta":`+delta+`,"finish_reason":`+finish+`}]}`+"\n\n")
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	llm, err := New(WithBaseURL(server.URL), WithToken("token"), WithModel("mistral-small-latest"))
	require.NoError(t, err)

	var streamedText, streamedThought strings.Builder
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "What is 17*23?"),
	}, llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
		switch chunk.Type {
		case streaming.ChunkTypeText:
			streamedText.WriteString(chunk.Content)
		case streaming.ChunkTypeReasoning:
			streamedThought.WriteString(chunk.Reasoning.Content)
		default:
		}
		return nil
	}))
	require.NoError(t, err)

	assert.Equal(t, "391", streamedText.String())
	assert.Equal(t, "17 times 23.", streamedThought.String())
	assert.Equal(t, "391", resp.Choices[0].Content)
	require.NotNil(t, resp.Choices[0].Reasoning)
	assert.Equal(t, "17 times 23.", resp.Choices[0].Reasoning.Content)
}

func TestAReasoningMistralModelGetsItsThinkingBackAsAChunkOfContent(t *testing.T) {
	t.Parallel()

	thinking := func(text string) map[string]any {
		return map[string]any{"type": "thinking", "thinking": []any{map[string]any{"type": "text", "text": text}}}
	}
	for _, model := range []string{
		"mistral-medium-latest", "mistral-small-latest", "magistral-medium-latest", "zai-glm-5-2", "mistral/mistral-small-latest",
	} {
		turns := assistantTurnsSent(t, model)
		require.Len(t, turns, 2, model)

		assert.Equal(t, []any{thinking("text turn thought"), map[string]any{"type": "text", "text": "answered in text"}},
			turns[0]["content"], model)
		assert.Equal(t, []any{thinking("tool turn thought")}, turns[1]["content"], model)
		assert.NotEmpty(t, turns[1]["tool_calls"], model)
		for _, turn := range turns {
			assert.NotContains(t, turn, "reasoning_content", model)
		}
	}
}

func TestAMistralModelThatDoesNotReasonGetsNoThinkingBack(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"codestral-latest", "mistral-large-latest", "ministral-8b-latest"} {
		turns := assistantTurnsSent(t, model)
		require.Len(t, turns, 2, model)

		assert.Equal(t, "answered in text", turns[0]["content"], model)
		assert.Empty(t, turns[1]["content"], model)
		assert.NotEmpty(t, turns[1]["tool_calls"], model)
		for _, turn := range turns {
			assert.NotContains(t, turn, "reasoning_content", model)
		}
	}
}
