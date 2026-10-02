package openai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestQwenIsSentAJSONSchemaOnlyWhereDashScopeDocumentsIt(t *testing.T) {
	t.Parallel()

	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "answer", Schema: json.RawMessage(
		`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)})
	call := func(baseURL, model string) (map[string]any, error) {
		doer := &bodyDoer{content: `{"a":"x"}`}
		llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, schema)
		if doer.body == nil {
			return nil, err
		}
		var body map[string]any
		require.NoError(t, json.Unmarshal(doer.body, &body))
		return body, err
	}

	const dashScope = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	for _, model := range []string{"qwen3.6-flash", "qwen-plus", "qwen3-max", "dashscope/qwen3.6-flash"} {
		body, err := call(dashScope, model)
		var unsupported *llms.ErrStructuredOutputUnsupported
		require.True(t, errors.As(err, &unsupported), "%s: %v", model, err)
		require.Nil(t, body, "%s: refused before the network", model)
	}

	for _, tc := range []struct{ baseURL, model string }{
		{dashScope, "qwen3.7-flash"}, {dashScope, "qwen3.8-max"}, {dashScope, "qwen3.7-plus"},
		{"http://litellm.internal/v1", "openrouter/qwen/qwen3.6-flash"},
	} {
		body, err := call(tc.baseURL, tc.model)
		require.NoError(t, err, tc.model)
		format, _ := body["response_format"].(map[string]any)
		assert.Equal(t, "json_schema", format["type"], tc.model)
	}
}
