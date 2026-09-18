package bedrock_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

const bigToolArgument = "10000000000000001"

func toolCallTurn(arguments string) []llms.MessageContent {
	return []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "book the room"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.ToolCall{
				ID:           "t1",
				Type:         "function",
				FunctionCall: &llms.FunctionCall{Name: "book", Arguments: arguments},
			},
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "t1", Name: "book", Content: "done"},
		}},
	}
}

const converseAnswer = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},` +
	`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`

const legacyAnswer = `{"id":"x","type":"message","role":"assistant","model":"m",` +
	`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",` +
	`"usage":{"input_tokens":1,"output_tokens":1}}`

func TestToolArgumentsKeepTheirPrecisionOnConverse(t *testing.T) {
	t.Parallel()

	llm, sent := legacyLLMCapturing(t, converseAnswer,
		bedrock.WithModel("amazon.nova-lite-v1:0"), bedrock.WithConverseAPI())

	_, err := llm.GenerateContent(context.Background(),
		toolCallTurn(`{"id":`+bigToolArgument+`}`))
	require.NoError(t, err)

	assert.Contains(t, *sent, `"id":`+bigToolArgument,
		"the integer the model chose reaches the vendor as a number, with every digit it had")
}

func TestToolArgumentsKeepTheirPrecisionOnTheLegacyDoor(t *testing.T) {
	t.Parallel()

	llm, sent := legacyLLMCapturing(t, legacyAnswer,
		bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"))

	_, err := llm.GenerateContent(context.Background(),
		toolCallTurn(`{"id":`+bigToolArgument+`}`))
	require.NoError(t, err)

	assert.Contains(t, *sent, `"id":`+bigToolArgument,
		"the legacy door carries the same integer as its Converse sibling")
}

func TestAFractionalToolArgumentStaysFractional(t *testing.T) {
	t.Parallel()

	llm, sent := legacyLLMCapturing(t, converseAnswer,
		bedrock.WithModel("amazon.nova-lite-v1:0"), bedrock.WithConverseAPI())

	_, err := llm.GenerateContent(context.Background(), toolCallTurn(`{"rate":1.5}`))
	require.NoError(t, err)

	assert.Contains(t, *sent, `"rate":1.5`, "a fraction must not be rounded into an integer")
}

func TestAToolArgumentInTheAnswerKeepsItsPrecision(t *testing.T) {
	t.Parallel()

	answer := `{"id":"x","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"tool_use","id":"t1","name":"book","input":{"id":` + bigToolArgument + `}}],` +
		`"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`

	llm, _ := legacyLLMCapturing(t, answer,
		bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"))

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "book the room")})
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	require.Len(t, resp.Choices[0].ToolCalls, 1)

	assert.Contains(t, resp.Choices[0].ToolCalls[0].FunctionCall.Arguments, `"id":`+bigToolArgument,
		"the integer the model chose reaches the caller as a number, with every digit it had")
}
