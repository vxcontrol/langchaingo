package bedrock_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type counterCase struct {
	name   string
	model  string
	whole  string
	chunks []string
}

func legacyCounterFamilies() []counterCase {
	return []counterCase{
		{
			name:  "amazon",
			model: "amazon.titan-text-express-v1",
			whole: `{"inputTextTokenCount":5,"results":[{"tokenCount":3,"outputText":"ok",` +
				`"completionReason":"FINISH"}]}`,
			chunks: []string{
				`{"outputText":"ok","index":0,"completionReason":"FINISH",` +
					`"inputTextTokenCount":5,"outputTextTokenCount":3}`,
			},
		},
		{
			name:  "meta",
			model: "meta.llama3-70b-instruct-v1:0",
			whole: `{"generation":"ok","prompt_token_count":5,"generation_token_count":3,"stop_reason":"stop"}`,
			chunks: []string{
				`{"generation":"ok","stop_reason":"stop","prompt_token_count":5,"generation_token_count":3}`,
			},
		},
		{
			name:  "nova",
			model: "amazon.nova-pro-v1:0",
			whole: `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},` +
				`"stopReason":"end_turn","usage":{"inputTokens":5,"outputTokens":3,"totalTokens":8}}`,
			chunks: []string{
				`{"messageStart":{"role":"assistant","usage":{"inputTokens":5}}}`,
				`{"contentBlockDelta":{"delta":{"text":"ok"}}}`,
				`{"messageDelta":{"stopReason":"end_turn","usage":{"outputTokens":3}}}`,
			},
		},
	}
}

func TestEveryLegacyFamilyReportsCountersAsInt(t *testing.T) {
	t.Parallel()

	counters := []string{"PromptTokens", "CompletionTokens", "TotalTokens"}

	for _, tc := range legacyCounterFamilies() {
		t.Run(tc.name+"/whole answer", func(t *testing.T) {
			t.Parallel()

			llm := truncationLLMWithBody(t, tc.whole, bedrock.WithModel(tc.model))
			resp, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
			require.NoError(t, err)
			require.Len(t, resp.Choices, 1)

			for _, key := range counters {
				_, ok := resp.Choices[0].GenerationInfo[key].(int)
				assert.True(t, ok, "%s must be an int, as the Claude doors report it, got %#v",
					key, resp.Choices[0].GenerationInfo[key])
			}
		})

		t.Run(tc.name+"/streamed", func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
				enc := eventstream.NewEncoder()
				for _, chunk := range tc.chunks {
					writeLegacyChunk(t, w, enc, chunk)
				}
			}))
			t.Cleanup(srv.Close)

			llm := bedrockLLMAgainst(t, srv, bedrock.WithModel(tc.model))
			resp, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
				llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
			require.NoError(t, err)
			require.Len(t, resp.Choices, 1)

			for _, key := range counters {
				value, present := resp.Choices[0].GenerationInfo[key]
				if !present {
					continue
				}
				_, ok := value.(int)
				assert.True(t, ok, "%s must be an int on the streamed path too, got %#v", key, value)
			}
		})
	}
}
