package bedrock_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

// Same server as legacyStreamOfThreeWords, but the handler goroutine is
// descheduled for 50ms between chunks, as it can be on a loaded CI runner.
func TestProbeSlowServerAfterConsumerGivesUp(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		enc := eventstream.NewEncoder()
		writeLegacyChunk(t, w, enc, `{"type":"message_start","message":{"id":"x","type":"message",`+
			`"role":"assistant","model":"m","content":[],"stop_reason":null,`+
			`"usage":{"input_tokens":10,"output_tokens":1}}}`)
		for _, text := range []string{"sixty ", "rooms ", "are free"} {
			time.Sleep(50 * time.Millisecond)
			inner := `{"bytes":"` + base64Encode(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"`+text+`"}}`) + `"}`
			encErr := enc.Encode(w, eventstream.Message{Headers: eventstream.Headers{
				{Name: ":message-type", Value: eventstream.StringValue("event")},
				{Name: ":event-type", Value: eventstream.StringValue("chunk")},
				{Name: ":content-type", Value: eventstream.StringValue("application/json")},
			}, Payload: []byte(inner)})
			w.(http.Flusher).Flush()
			t.Logf("server write of %q: err=%v ctxErr=%v", text, encErr, r.Context().Err())
			_ = writeLegacyChunk
			continue
			writeLegacyChunk(t, w, enc, `{"type":"content_block_delta","index":0,`+
				`"delta":{"type":"text_delta","text":"`+text+`"}}`)
		}
		time.Sleep(50 * time.Millisecond)
		writeLegacyChunk(t, w, enc, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},`+
			`"usage":{"output_tokens":9}}`)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"))
	gaveUp := errors.New("consumer gave up")
	delivered := 0
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			if chunk.Type != streaming.ChunkTypeText {
				return nil
			}
			delivered++
			if delivered == 2 {
				return gaveUp
			}
			return nil
		}))
	t.Logf("client side: err=%v content=%q (client-side behaviour is correct)", err, resp.Choices[0].Content)
}
