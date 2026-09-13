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

	thought := func(text string) *reasoning.ContentReasoning { return &reasoning.ContentReasoning{Content: text} }
	history := []llms.MessageContent{
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
	}
	tools := []llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "f", Parameters: map[string]any{"type": "object"},
	}}}

	_, err = llm.GenerateContent(context.Background(), history, llms.WithTools(tools))
	require.NoError(t, err)

	var replayed []any
	for _, raw := range body["messages"].([]any) {
		msg := raw.(map[string]any)
		if msg["role"] == "assistant" {
			replayed = append(replayed, msg["reasoning_content"])
		}
	}
	return replayed
}

func TestPreservedThinkingGetsBackTheReasoningOfEveryAssistantTurn(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"deepseek-v4-pro", "deepseek-flash", "deepseek/deepseek-v4-pro",
		"kimi-k3", "kimi-k2.7-code", "moonshot/kimi-k2.6", "glm-5.2", "zai/glm-5.3",
	} {
		assert.Equal(t, []any{"text turn thought", "tool turn thought"}, replayedReasoning(t, model), model)
	}
}

func TestOtherVendorsGetBackOnlyTheReasoningOfToolTurns(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"qwen3.7-plus", "grok-4", "mistral-medium-latest"} {
		assert.Equal(t, []any{nil, "tool turn thought"}, replayedReasoning(t, model), model)
	}
}
