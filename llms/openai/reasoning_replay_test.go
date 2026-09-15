package openai

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
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func replayedReasoning(t *testing.T, model string) []any {
	t.Helper()

	turns := assistantTurnsSent(t, model)
	replayed := make([]any, 0, len(turns))
	for _, turn := range turns {
		replayed = append(replayed, turn["reasoning_content"])
	}
	return replayed
}

func assistantTurnsSent(t *testing.T, model string) []map[string]any {
	t.Helper()

	thought := func(text string) *reasoning.ContentReasoning { return &reasoning.ContentReasoning{Content: text} }
	return assistantTurnsSentFor(t, model, []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "first"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextPartWithReasoning("answered in text", thought("text turn thought")),
		}},
		llms.TextParts(llms.ChatMessageTypeHuman, "second"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextPartWithReasoning("", thought("tool turn thought")),
			llms.ToolCall{ID: "c1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "f", Arguments: "{}"}},
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "c1", Name: "f", Content: "done"},
		}},
	})
}

func assistantTurnsSentFor(t *testing.T, model string, history []llms.MessageContent) []map[string]any {
	t.Helper()

	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer server.Close()

	llm, err := New(WithBaseURL(server.URL), WithToken("token"), WithModel(model), WithPreserveReasoningContent())
	require.NoError(t, err)

	tools := []llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "f", Parameters: map[string]any{"type": "object"},
	}}}

	_, err = llm.GenerateContent(context.Background(), history, llms.WithTools(tools))
	require.NoError(t, err)

	var turns []map[string]any
	for _, raw := range body["messages"].([]any) {
		msg := raw.(map[string]any)
		if msg["role"] == "assistant" {
			turns = append(turns, msg)
		}
	}
	return turns
}

func TestPreservedThinkingGetsBackTheReasoningOfEveryAssistantTurn(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"deepseek-v4-pro", "deepseek-flash", "deepseek/deepseek-v4-pro",
		"kimi-k3", "kimi-k2.7-code", "moonshot/kimi-k2.6", "glm-5.2", "zai/glm-5.3",
		"qwen3.7-plus", "dashscope/qwen3.8-max",
	} {
		assert.Equal(t, []any{"text turn thought", "tool turn thought"}, replayedReasoning(t, model), model)
	}
}

func TestMiniMaxGetsBackTheReasoningOfEveryAssistantTurnInThinkTags(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"minimax/MiniMax-M3", "MiniMax-M2.7"} {
		turns := assistantTurnsSent(t, model)
		require.Len(t, turns, 2, model)

		assert.Equal(t, "<think>text turn thought</think>answered in text", turns[0]["content"], model)
		assert.Equal(t, "<think>tool turn thought</think>", turns[1]["content"], model)
		assert.NotEmpty(t, turns[1]["tool_calls"], model)
		for _, turn := range turns {
			assert.NotContains(t, turn, "reasoning_content", model)
		}
	}
}

func TestMiniMaxTurnThatAlreadyHoldsItsThinkBlockGoesBackAsItIs(t *testing.T) {
	t.Parallel()

	const textThought = `The user is asking me to reply with the single word "PONG".`
	const toolThought = `The user is asking me to call the get_weather function for Paris.`
	textContent := "<think>\n" + textThought + "\n</think>\n\nPONG"
	toolContent := "<think>\n" + toolThought + "\n</think>\n\nI'll get the current weather for Paris for you."

	for _, model := range []string{"minimax/MiniMax-M3", "MiniMax-M2.7"} {
		turns := assistantTurnsSentFor(t, model, []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Reply with the single word PONG."),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.TextPartWithReasoning(textContent, &reasoning.ContentReasoning{Content: textThought}),
			}},
			llms.TextParts(llms.ChatMessageTypeHuman, "Call get_weather for Paris."),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.TextPartWithReasoning(toolContent, &reasoning.ContentReasoning{Content: toolThought}),
				llms.ToolCall{ID: "c1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "f", Arguments: "{}"}},
			}},
			{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
				llms.ToolCallResponse{ToolCallID: "c1", Name: "f", Content: "done"},
			}},
		})
		require.Len(t, turns, 2, model)

		assert.Equal(t, textContent, turns[0]["content"], model)
		assert.Equal(t, toolContent, turns[1]["content"], model)
		for _, turn := range turns {
			assert.NotContains(t, turn, "reasoning_content", model)
		}
	}
}

func TestOtherVendorsGetBackOnlyTheReasoningOfToolTurns(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"grok-4", "mistralai/mistral-small-2603", "openrouter/mistralai/mistral-medium-3-5", "magistral:24b",
		"mistral-ai/mistral-small-2603",
	} {
		assert.Equal(t, []any{nil, "tool turn thought"}, replayedReasoning(t, model), model)
	}
}
