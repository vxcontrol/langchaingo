package openai

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestAResponsesOnlyModelIsRefusedBeforeOpenAIsChatCompletions(t *testing.T) {
	t.Parallel()

	for _, baseURL := range []string{"", "https://api.openai.com/v1", "https://eu.api.openai.com/v1"} {
		for _, model := range []string{"gpt-5.6-cyber", "gpt-daybreak-red-latest", "gpt-daybreak-blue-latest"} {
			doer := &bodyDoer{}
			opts := []Option{WithModel(model), WithHTTPClient(doer)}
			if baseURL != "" {
				opts = append(opts, WithBaseURL(baseURL))
			}
			_, err := newUnitLLM(t, opts...).GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
			var refused *reasoning.ErrChatCompletionsUnsupported
			require.True(t, errors.As(err, &refused), "%q %s: %v", baseURL, model, err)
			require.Nil(t, doer.body, "%q %s: refused before the network", baseURL, model)
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
