package openai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAResponsesOnlyModelGoesToOpenAIsResponses(t *testing.T) {
	t.Parallel()

	for _, baseURL := range []string{"", "https://api.openai.com/v1", "https://eu.api.openai.com/v1"} {
		for _, model := range []string{"gpt-5.6-cyber", "gpt-daybreak-red-latest", "gpt-daybreak-blue-latest"} {
			doer := &bodyDoer{}
			opts := []Option{WithModel(model), WithHTTPClient(doer)}
			if baseURL != "" {
				opts = append(opts, WithBaseURL(baseURL))
			}
			resp, err := newUnitLLM(t, opts...).GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
			require.NoError(t, err, "%q %s", baseURL, model)
			require.Equal(t, "/v1/responses", doer.path, "%q %s", baseURL, model)
			var body map[string]any
			require.NoError(t, json.Unmarshal(doer.body, &body))
			require.Equal(t, []any{map[string]any{"type": "message", "role": "user", "content": "hi"}}, body["input"])
			require.Equal(t, "ok", resp.Choices[0].Content)
		}
	}

	for _, tc := range []struct{ baseURL, model string }{
		{"http://litellm.internal/v1", "openai/gpt-5.6-cyber"},
		{"https://api.openai.com/v1", "gpt-5.6"},
	} {
		body, _ := hostCall(t, tc.baseURL, tc.model)
		assert.Equal(t, tc.model, body["model"], "%s on %s goes out", tc.model, tc.baseURL)
		assert.Equal(t, []any{map[string]any{"role": "user", "content": "hi"}}, body["messages"], tc.model)
	}
}
