package openai

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestClient(t *testing.T, opts ...Option) *LLM {
	t.Helper()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
		return nil
	}

	llm, err := New(opts...)
	require.NoError(t, err)
	return llm
}

func TestMultiContentText(t *testing.T) {
	t.Parallel()

	llm := newTestClient(t)

	parts := []llms.ContentPart{
		llms.TextPart("I'm a pomeranian"),
		llms.TextPart("What kind of mammal am I?"),
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	rsp, err := llm.GenerateContent(t.Context(), content)
	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]
	assert.Regexp(t, "dog|canid", strings.ToLower(c1.Content))
}

func TestMultiContentTextChatSequence(t *testing.T) {
	t.Parallel()

	llm := newTestClient(t)

	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart("Name some countries")},
		},
		{
			Role:  llms.ChatMessageTypeAI,
			Parts: []llms.ContentPart{llms.TextPart("Spain and Lesotho")},
		},
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart("Which if these is larger?")},
		},
	}

	rsp, err := llm.GenerateContent(t.Context(), content)
	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]
	assert.Regexp(t, "spain.*larger", strings.ToLower(c1.Content))
}

func TestMultiContentImage(t *testing.T) {
	t.Parallel()

	llm := newTestClient(t, WithModel("gpt-4o"))

	parts := []llms.ContentPart{
		llms.ImageURLPart("https://github.com/vxcontrol/langchaingo/blob/main/docs/static/img/parrot-icon.png?raw=true"), //nolint:lll
		llms.TextPart("describe this image in detail"),
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	rsp, err := llm.GenerateContent(t.Context(), content, llms.WithMaxTokens(300))
	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]
	assert.Regexp(t, "parrot", strings.ToLower(c1.Content))
}

func TestWithStreaming(t *testing.T) {
	t.Parallel()

	llm := newTestClient(t)

	parts := []llms.ContentPart{
		llms.TextPart("I'm a pomeranian"),
		llms.TextPart("Tell me more about my taxonomy"),
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	var sb strings.Builder
	rsp, err := llm.GenerateContent(t.Context(), content,
		llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			sb.Write(chunk)
			return nil
		}))

	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]
	assert.Regexp(t, "dog|canid", strings.ToLower(c1.Content))
	assert.Regexp(t, "dog|canid", strings.ToLower(sb.String()))
}

//nolint:lll
func TestFunctionCall(t *testing.T) {
	t.Parallel()

	llm := newTestClient(t)

	parts := []llms.ContentPart{
		llms.TextPart("What is the weather like in Boston?"),
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	functions := []llms.FunctionDefinition{
		{
			Name:        "getCurrentWeather",
			Description: "Get the current weather in a given location",
			Parameters:  json.RawMessage(`{"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}`), //nolint:lll
		},
	}

	rsp, err := llm.GenerateContent(t.Context(), content, llms.WithFunctions(functions))
	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]
	assert.Equal(t, "tool_calls", c1.StopReason)
	assert.NotNil(t, c1.FuncCall)
}
