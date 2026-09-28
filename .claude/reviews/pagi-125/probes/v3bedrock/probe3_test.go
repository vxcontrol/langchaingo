package bedrock_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func TestProbeV3CounterTypes(t *testing.T) {
	cases := legacyCounterFamilies()
	cases = append(cases, counterCase{name: "claude", model: "anthropic.claude-sonnet-4-5-20250929-v1:0",
		whole: `{"id":"x","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}`})
	for _, tc := range cases {
		llm := truncationLLMWithBody(t, tc.whole, bedrock.WithModel(tc.model))
		r, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
		if err != nil {
			t.Logf("%s whole err %v", tc.name, err)
			continue
		}
		gi := r.Choices[0].GenerationInfo
		v, ok := gi["input_tokens"].(int)
		t.Logf("%-7s whole:    input_tokens %T output_tokens %T  .(int)=%d,%v", tc.name, gi["input_tokens"], gi["output_tokens"], v, ok)
		if tc.chunks == nil {
			continue
		}
		chunks := tc.chunks
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
			enc := eventstream.NewEncoder()
			for _, c := range chunks {
				writeLegacyChunk(t, w, enc, c)
			}
		}))
		llm = bedrockLLMAgainst(t, srv, bedrock.WithModel(tc.model))
		r, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
		srv.Close()
		if err != nil {
			t.Logf("%s stream err %v", tc.name, err)
			continue
		}
		gi = r.Choices[0].GenerationInfo
		v, ok = gi["input_tokens"].(int)
		t.Logf("%-7s streamed: input_tokens %T output_tokens %T  .(int)=%d,%v  %s", tc.name, gi["input_tokens"], gi["output_tokens"], v, ok, fmt.Sprint(gi["PromptTokens"]))
	}
}
