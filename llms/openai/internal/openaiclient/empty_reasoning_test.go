package openaiclient

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAKeptEmptyReasoningGoesOutAsAnEmptyString(t *testing.T) {
	t.Parallel()

	toolCall := []ToolCall{{ID: "c1", Type: ToolTypeFunction, Function: ToolFunction{Name: "f", Arguments: "{}"}}}
	for name, tc := range map[string]struct {
		msg  ChatMessage
		want string
	}{
		"text": {
			ChatMessage{Role: "assistant", Content: "hi", KeepsEmptyReasoning: true},
			`{"role":"assistant","content":"hi","reasoning_content":""}`,
		},
		"tool call": {
			ChatMessage{Role: "assistant", ToolCalls: toolCall, KeepsEmptyReasoning: true},
			`{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function",` +
				`"function":{"name":"f","arguments":"{}"}}],"reasoning_content":""}`,
		},
		"content parts": {
			ChatMessage{Role: "assistant", MultiContent: []llms.ContentPart{
				llms.TextContent{Text: "a"}, llms.TextContent{Text: "b"},
			}, ToolCalls: toolCall, KeepsEmptyReasoning: true},
			`{"role":"assistant","content":[{"text":"a","type":"text"},{"text":"b","type":"text"}],` +
				`"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}],` +
				`"reasoning_content":""}`,
		},
	} {
		raw, err := json.Marshal(tc.msg)
		require.NoError(t, err, name)
		assert.Equal(t, tc.want, string(raw), name)
	}
}

func TestAKeptReasoningThatHasTextGoesOutOnce(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		msg  ChatMessage
		want string
	}{
		"reasoning_content": {
			ChatMessage{Role: "assistant", Content: "hi", ReasoningContent: "thought", KeepsEmptyReasoning: true},
			`{"role":"assistant","content":"hi","reasoning_content":"thought"}`,
		},
		"reasoning": {
			ChatMessage{Role: "assistant", Content: "hi", Reasoning: "thought", KeepsEmptyReasoning: true},
			`{"role":"assistant","content":"hi","reasoning_content":"thought","reasoning":"thought"}`,
		},
	} {
		raw, err := json.Marshal(tc.msg)
		require.NoError(t, err, name)
		assert.Equal(t, tc.want, string(raw), name)
	}
}

func TestAnEmptyReasoningStaysOffTheWireUnlessKept(t *testing.T) {
	t.Parallel()

	for name, msg := range map[string]ChatMessage{
		"text":          {Role: "assistant", Content: "hi"},
		"content parts": {Role: "assistant", MultiContent: []llms.ContentPart{llms.TextContent{Text: "a"}, llms.TextContent{Text: "b"}}},
		"user":          {Role: "user", Content: "hi"},
	} {
		raw, err := json.Marshal(msg)
		require.NoError(t, err, name)
		assert.NotContains(t, string(raw), "reasoning_content", name)
	}
}
