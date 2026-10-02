package bedrock_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func converseTurnsOnTheWire(t *testing.T, model string, history []llms.MessageContent) []string {
	t.Helper()

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, converseAnswer)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, bedrock.WithModel(model), bedrock.WithConverseAPI())
	_, err := llm.GenerateContent(t.Context(), history, llms.WithTools(lookupTools()))
	require.NoError(t, err, model)

	var sent struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(raw, &sent), model)
	turns := make([]string, 0, len(sent.Messages))
	for _, message := range sent.Messages {
		turn := message.Role
		for _, block := range message.Content {
			for kind := range block {
				turn += " " + kind
			}
		}
		turns = append(turns, turn)
	}
	return turns
}

func TestConverseSendsToolResultsAndTheNextHumanMessageAsOneUserTurn(t *testing.T) {
	t.Parallel()

	lookup := llms.ToolCall{ID: "call-1", Type: "function",
		FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: `{"q":"x"}`}}
	result := llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
		llms.ToolCallResponse{ToolCallID: "call-1", Name: "lookup", Content: "found"}}}
	ask := llms.TextParts(llms.ChatMessageTypeHuman, "look it up")
	nudge := llms.TextParts(llms.ChatMessageTypeHuman, "the deadline is near")
	want := []string{"user text", "assistant toolUse", "user toolResult text"}

	require.Equal(t, want, converseTurnsOnTheWire(t, converseClaude, []llms.MessageContent{
		ask, {Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{lookup}}, result, nudge,
	}))
	require.Equal(t, want, converseTurnsOnTheWire(t, converseClaude, []llms.MessageContent{
		ask, {Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{lookup}}, result,
		llms.TextParts(llms.ChatMessageTypeSystem, "be brief"), nudge,
	}))

	thought := &reasoning.ContentReasoning{Content: "look it up first", Signature: []byte("sig")}
	k3 := converseTurnsOnTheWire(t, "us.moonshotai.kimi-k3", []llms.MessageContent{
		ask,
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("", thought), lookup}},
		result,
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("", thought)}},
		nudge,
	})
	require.Equal(t, want, k3, "an answer of only reasoning is dropped for Kimi K3, so the tool results meet the nudge")
}
