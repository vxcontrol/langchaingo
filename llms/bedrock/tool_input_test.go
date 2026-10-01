package bedrock_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestAReplayedClaudeCallWithoutArgumentsSendsAnEmptyInputObject(t *testing.T) {
	t.Parallel()

	for _, arguments := range []string{"null", "{}", "", "  "} {
		t.Run("arguments "+arguments, func(t *testing.T) {
			t.Parallel()

			llm, sent := legacyLLMCapturing(t, legacyAnswer,
				bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"))

			_, err := llm.GenerateContent(context.Background(), toolCallTurn(arguments))
			require.NoError(t, err)

			var body struct {
				Messages []struct {
					Content []map[string]json.RawMessage `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal([]byte(*sent), &body))
			require.Len(t, body.Messages, 3)
			toolUse := body.Messages[1].Content[0]
			require.JSONEq(t, `"tool_use"`, string(toolUse["type"]))
			require.Contains(t, toolUse, "input",
				"the Messages API on Bedrock requires input on every tool_use block")
			assert.JSONEq(t, `{}`, string(toolUse["input"]))
		})
	}
}

func TestACachedClaudeCallWithoutArgumentsKeepsItsOtherFields(t *testing.T) {
	t.Parallel()

	rec := &legacyRecorder{responses: []string{legacyAnswer}}
	_, err := rec.serve(t).GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "what time is it"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.ToolCall{ID: "A", Type: "function", FunctionCall: &llms.FunctionCall{Name: "get_time", Arguments: "{}"}},
			bedrock.WithCacheControl(llms.TextContent{}, bedrock.EphemeralCache()),
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "A", Name: "get_time", Content: "12:34"},
		}},
	})
	require.NoError(t, err)

	var body struct {
		Messages []struct {
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(rec.requests[0], &body))
	require.Len(t, body.Messages, 3)
	require.Len(t, body.Messages[1].Content, 1)
	assert.JSONEq(t,
		`{"type":"tool_use","id":"A","name":"get_time","input":{},"cache_control":{"type":"ephemeral","ttl":"5m"}}`,
		string(body.Messages[1].Content[0]))
}

func TestAReplayedCallWithoutArgumentsSendsAnEmptyInputOnConverse(t *testing.T) {
	t.Parallel()

	for _, arguments := range []string{"null", "{}", "", "  "} {
		t.Run("arguments "+arguments, func(t *testing.T) {
			t.Parallel()

			llm, sent := legacyLLMCapturing(t, converseAnswer,
				bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())

			_, err := llm.GenerateContent(context.Background(), toolCallTurn(arguments))
			require.NoError(t, err)

			var body struct {
				Messages []struct {
					Content []struct {
						ToolUse map[string]json.RawMessage `json:"toolUse"`
					} `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal([]byte(*sent), &body))
			require.Len(t, body.Messages, 3)
			toolUse := body.Messages[1].Content[0].ToolUse
			require.NotNil(t, toolUse)
			assert.JSONEq(t, `{}`, string(toolUse["input"]))
		})
	}
}
