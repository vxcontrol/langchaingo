package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func TestReasoningCountsAsOutputWhereverTheHostReportsIt(t *testing.T) {
	t.Parallel()

	for name, usage := range map[string]struct {
		json       string
		completion int
	}{
		"outside completion_tokens, as xAI reports it": {
			`{"prompt_tokens":335,"completion_tokens":14,"total_tokens":374,` +
				`"completion_tokens_details":{"reasoning_tokens":25}}`, 39,
		},
		"inside completion_tokens, as OpenAI reports it": {
			`{"prompt_tokens":335,"completion_tokens":39,"total_tokens":374,` +
				`"completion_tokens_details":{"reasoning_tokens":25}}`, 39,
		},
		"no reasoning": {`{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}`, 5},
	} {
		for _, streamed := range []bool{false, true} {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if streamed {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"grok-4.5",`+
						`"choices":[{"index":0,"delta":{"role":"assistant","content":"4"},"finish_reason":"stop"}]}`+"\n\n"+
						`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"grok-4.5","choices":[],`+
						`"usage":`+usage.json+`}`+"\n\n"+"data: [DONE]\n\n")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"grok-4.5",`+
					`"choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}],`+
					`"usage":`+usage.json+`}`)
			}))
			t.Cleanup(srv.Close)

			opts := []llms.CallOption{}
			if streamed {
				opts = append(opts, llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
			}
			llm := newUnitLLM(t, WithBaseURL(srv.URL), WithModel("grok-4.5"))
			resp, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "2+2?")}, opts...)
			require.NoError(t, err, name)
			info := resp.Choices[0].GenerationInfo
			require.Equal(t, usage.completion, info["CompletionTokens"], "%s streamed=%v", name, streamed)
			require.Equal(t, info["TotalTokens"], info["PromptTokens"].(int)+info["CompletionTokens"].(int),
				"%s streamed=%v", name, streamed)
		}
	}
}
