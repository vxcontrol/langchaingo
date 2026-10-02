package openai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestTheEffortNoneTurnsReasoningOffOnTheWire(t *testing.T) {
	t.Parallel()

	none := llms.WithReasoning(llms.ReasoningEffort(reasoning.OpenAIDisableEffort), 0)
	send := func(baseURL, model string, opts ...llms.CallOption) (map[string]any, error) {
		doer := &bodyDoer{}
		llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
		if doer.body == nil {
			return nil, err
		}
		var body map[string]any
		require.NoError(t, json.Unmarshal(doer.body, &body))
		return body, err
	}

	body, err := send("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen3.6-flash", none)
	require.NoError(t, err)
	assert.Equal(t, false, body["enable_thinking"])

	body, err = send("", "gpt-5.4", none, llms.WithTools([]llms.Tool{{Type: "function",
		Function: &llms.FunctionDefinition{Name: "lookup", Parameters: map[string]any{"type": "object"}}}}))
	require.NoError(t, err, "tools go out on Chat Completions with the effort none")
	assert.Equal(t, "none", body["reasoning_effort"])

	body, err = send("https://api.x.ai/v1", "grok-4.7", none)
	var offUnsupported *reasoning.ErrReasoningOffUnsupported
	require.True(t, errors.As(err, &offUnsupported), "got %v", err)
	assert.Nil(t, body, "the refusal comes before the network")
}
