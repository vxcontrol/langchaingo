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
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

// loopWithoutReasoning is the history a caller replays after DeepSeek streamed an
// empty reasoning_content: no assistant turn holds any reasoning, and the tool turn
// holds nothing but its call.
func loopWithoutReasoning() []llms.MessageContent {
	return []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "first"),
		llms.TextParts(llms.ChatMessageTypeAI, "answered in text"),
		llms.TextParts(llms.ChatMessageTypeHuman, "second"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.ToolCall{ID: "c1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "f", Arguments: "{}"}},
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "c1", Name: "f", Content: "done"},
		}},
	}
}

func TestDeepSeekTurnWithoutReasoningGoesBackWithAnEmptyOne(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"deepseek-flash", "deepseek-v4-pro", "DeepSeek-V4-Pro", "deepseek/deepseek-v4-pro",
		"deepseek-chat", "deepseek-reasoner",
	} {
		turns := assistantTurnsSentFor(t, model, loopWithoutReasoning())
		require.Len(t, turns, 2, model)

		for _, turn := range turns {
			require.Contains(t, turn, "reasoning_content", model)
			assert.Equal(t, "", turn["reasoning_content"], model)
		}
		assert.NotEmpty(t, turns[1]["tool_calls"], model)
	}
}

func TestDeepSeekTurnWithAnEmptyReasoningPartGoesBackWithAnEmptyOne(t *testing.T) {
	t.Parallel()

	turns := assistantTurnsSentFor(t, "deepseek-flash", []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "first"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextPartWithReasoning("", &reasoning.ContentReasoning{}),
			llms.ToolCall{ID: "c1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "f", Arguments: "{}"}},
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "c1", Name: "f", Content: "done"},
		}},
	})
	require.Len(t, turns, 1)

	require.Contains(t, turns[0], "reasoning_content")
	assert.Equal(t, "", turns[0]["reasoning_content"])
}

func TestOnlyDeepSeekGetsAnEmptyReasoningBack(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"kimi-k3", "moonshot/kimi-k2.6", "glm-5.3", "zai/glm-5.3", "qwen3.7-plus", "dashscope/qwen3.8-max",
		"grok-4", "gpt-5.5", "mistralai/mistral-small-2603", "MiniMax-M2.7", "deepseekcoder",
	} {
		for _, turn := range assistantTurnsSentFor(t, model, loopWithoutReasoning()) {
			assert.NotContains(t, turn, "reasoning_content", model)
		}
	}
}

func TestDeepSeekWithoutPreservedReasoningSendsNoReasoningField(t *testing.T) {
	t.Parallel()

	var raw string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		raw = string(body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer server.Close()

	llm, err := New(WithBaseURL(server.URL), WithToken("token"), WithModel("deepseek-flash"))
	require.NoError(t, err)
	_, err = llm.GenerateContent(context.Background(), loopWithoutReasoning())
	require.NoError(t, err)

	assert.False(t, strings.Contains(raw, "reasoning_content"), raw)
}
