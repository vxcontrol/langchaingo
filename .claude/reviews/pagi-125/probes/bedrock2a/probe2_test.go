package bedrock_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func TestProbeCounterTypes(t *testing.T) {
	cases := []struct {
		model, whole string
		chunks       []string
	}{
		{"ai21.jamba-1-5-large-v1:0",
			`{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			[]string{`{"text":"ok","finish_reason":"stop","index":0,"usage":{"prompt_tokens":5,"completion_tokens":3}}`}},
		{"us.amazon.nova-pro-v1:0",
			`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":5,"outputTokens":3,"totalTokens":8}}`,
			[]string{`{"contentBlockDelta":{"delta":{"text":"ok"},"contentBlockIndex":0}}`, `{"messageStop":{"stopReason":"end_turn"}}`, `{"metadata":{"usage":{"inputTokens":5,"outputTokens":3}}}`}},
		{"anthropic.claude-sonnet-4-5-20250929-v1:0",
			`{"id":"x","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}`,
			nil},
	}
	for _, tc := range cases {
		llm := truncationLLMWithBody(t, tc.whole, bedrock.WithModel(tc.model))
		resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
		if err != nil {
			t.Fatal(err)
		}
		gi := resp.Choices[0].GenerationInfo
		t.Logf("%s whole: input_tokens %T, PromptTokens %T", tc.model, gi["input_tokens"], gi["PromptTokens"])
		if tc.chunks == nil {
			continue
		}
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
			enc := eventstream.NewEncoder()
			for _, c := range tc.chunks {
				writeLegacyChunk(t, w, enc, c)
			}
		}))
		llm2 := bedrockLLMAgainst(t, srv, bedrock.WithModel(tc.model))
		resp, err = llm2.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
		srv.Close()
		if err != nil {
			t.Fatal(err)
		}
		gi = resp.Choices[0].GenerationInfo
		t.Logf("%s streamed: input_tokens %T, PromptTokens %T", tc.model, gi["input_tokens"], gi["PromptTokens"])
	}
}
