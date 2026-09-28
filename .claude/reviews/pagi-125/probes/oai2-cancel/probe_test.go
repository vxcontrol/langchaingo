package openai_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func TestProbeCancelMidStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 50; i++ {
			_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"w`+fmt.Sprint(i)+` "},"finish_reason":null}]}`+"\n\n")
			w.(http.Flusher).Flush()
			select {
			case <-r.Context().Done():
				return
			case <-time.After(20 * time.Millisecond):
			}
		}
		_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	llm, err := openai.New(openai.WithToken("k"), openai.WithBaseURL(srv.URL), openai.WithModel("gpt-4o"))
	if err != nil {
		t.Fatal(err)
	}
	nilErr, withErr := 0, 0
	for trial := 0; trial < 20; trial++ {
		ctx, cancel := context.WithCancel(context.Background())
		n := 0
		resp, err := llm.GenerateContent(ctx, []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithStreamingFunc(func(_ context.Context, c streaming.Chunk) error {
				if c.Type == streaming.ChunkTypeText {
					n++
					if n == 3 {
						cancel()
					}
				}
				return nil
			}))
		cancel()
		if err == nil {
			nilErr++
			if trial < 3 {
				fmt.Printf("trial %d: err=nil content=%q stop=%q\n", trial, resp.Choices[0].Content, resp.Choices[0].StopReason)
			}
		} else {
			withErr++
			if trial < 3 {
				fmt.Printf("trial %d: err=%v resp=%v\n", trial, err, resp != nil)
			}
		}
	}
	fmt.Printf("cancelled mid-stream: nil error %d/20, error %d/20\n", nilErr, withErr)
}
