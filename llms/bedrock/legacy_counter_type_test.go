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
	stop   string
}

var wantCounters = map[string]int{"PromptTokens": 5, "CompletionTokens": 3, "TotalTokens": 8}

func legacyCounterFamilies() []counterCase {
	return []counterCase{
		{
			name:  "amazon",
			stop:  "FINISH",
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
			stop:  "stop",
			model: "meta.llama3-70b-instruct-v1:0",
			whole: `{"generation":"ok","prompt_token_count":5,"generation_token_count":3,"stop_reason":"stop"}`,
			chunks: []string{
				`{"generation":"o","prompt_token_count":5,"generation_token_count":1,"stop_reason":null}`,
				`{"generation":"k","prompt_token_count":null,"generation_token_count":3,"stop_reason":"stop"}`,
			},
		},
		{
			name:  "ai21",
			stop:  "stop",
			model: "ai21.jamba-1-5-large-v1:0",
			whole: `{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},` +
				`"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,` +
				`"total_tokens":8},"model":"ai21.jamba-1-5-large-v1:0"}`,
			chunks: []string{
				`{"text":"ok","finish_reason":"stop","index":0,` +
					`"usage":{"prompt_tokens":5,"completion_tokens":3}}`,
			},
		},
		{
			name:  "nova",
			stop:  "end_turn",
			model: "amazon.nova-pro-v1:0",
			whole: `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},` +
				`"stopReason":"end_turn","usage":{"inputTokens":5,"outputTokens":3,"totalTokens":8}}`,
			chunks: []string{
				`{"messageStart":{"role":"assistant"}}`,
				`{"contentBlockDelta":{"delta":{"text":"ok"},"contentBlockIndex":0}}`,
				`{"contentBlockStop":{"contentBlockIndex":0}}`,
				`{"messageStop":{"stopReason":"end_turn"}}`,
				`{"metadata":{"usage":{"inputTokens":5,"outputTokens":3,` +
					`"cacheReadInputTokenCount":0,"cacheWriteInputTokenCount":0},"metrics":{}}}`,
			},
		},
	}
}

func TestEveryLegacyFamilyReportsCountersAsInt(t *testing.T) {
	t.Parallel()

	for _, tc := range legacyCounterFamilies() {
		t.Run(tc.name+"/whole answer", func(t *testing.T) {
			t.Parallel()

			llm := truncationLLMWithBody(t, tc.whole, bedrock.WithModel(tc.model))
			resp, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
			require.NoError(t, err)
			require.Len(t, resp.Choices, 1)

			assertCounters(t, resp.Choices[0].GenerationInfo)
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

			assertCounters(t, resp.Choices[0].GenerationInfo)
			assert.Equal(t, tc.stop, resp.Choices[0].StopReason,
				"the reason the vendor gave for stopping reaches the caller on the streamed path too")
		})
	}
}

func assertCounters(t *testing.T, info map[string]any) {
	t.Helper()

	for key, want := range wantCounters {
		value, present := info[key]
		if !assert.True(t, present, "%s is missing from the generation info", key) {
			continue
		}
		got, ok := value.(int)
		if !assert.True(t, ok, "%s must be an int, as the Claude doors report it, got %#v", key, value) {
			continue
		}
		assert.Equal(t, want, got, "%s carries the number the vendor sent", key)
	}
}
