package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type countingDoer struct{ calls int }

func (d *countingDoer) Do(*http.Request) (*http.Response, error) {
	d.calls++
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(`{"id":"x","object":"chat.completion","created":1,"model":"m",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)),
	}, nil
}

func TestAResponsesOnlyModelIsRefusedBeforeOpenAIsChatCompletions(t *testing.T) {
	t.Parallel()

	call := func(baseURL, model string) (int, error) {
		doer := &countingDoer{}
		opts := []Option{WithModel(model), WithHTTPClient(doer)}
		if baseURL != "" {
			opts = append(opts, WithBaseURL(baseURL))
		}
		llm := newUnitLLM(t, opts...)
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
		return doer.calls, err
	}

	for _, baseURL := range []string{"", "https://api.openai.com/v1", "https://eu.api.openai.com/v1"} {
		for _, model := range []string{"gpt-5.6-cyber", "gpt-daybreak-red-latest", "gpt-daybreak-blue-latest"} {
			calls, err := call(baseURL, model)
			var refused *reasoning.ErrChatCompletionsUnsupported
			require.True(t, errors.As(err, &refused), "%q %s: %v", baseURL, model, err)
			require.Zero(t, calls, "%q %s: refused before the network", baseURL, model)
		}
	}

	for _, tc := range []struct{ baseURL, model string }{
		{"http://litellm.internal/v1", "openai/gpt-5.6-cyber"},
		{"https://api.openai.com/v1", "gpt-5.6"},
	} {
		calls, err := call(tc.baseURL, tc.model)
		require.NoError(t, err, tc.model)
		require.Equal(t, 1, calls, "%s on %s goes out", tc.model, tc.baseURL)
	}
}
