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

func TestProbeVerF3(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 40; i++ {
			_, _ = fmt.Fprintf(w, "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"w%d \"},\"finish_reason\":null}]}\n\n", i)
			w.(http.Flusher).Flush()
			select {
			case <-r.Context().Done():
				return
			case <-time.After(15 * time.Millisecond):
			}
		}
		_, _ = io.WriteString(w, "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	llm, err := openai.New(openai.WithToken("k"), openai.WithBaseURL(srv.URL), openai.WithModel("gpt-4o"))
	if err != nil {
		t.Fatal(err)
	}
	msgs := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
	for _, mode := range []string{"cancel-in-callback", "deadline"} {
		nilErr, withErr := 0, 0
		shown := false
		for trial := 0; trial < 30; trial++ {
			var ctx context.Context
			var cancel context.CancelFunc
			if mode == "deadline" {
				ctx, cancel = context.WithTimeout(context.Background(), 80*time.Millisecond)
			} else {
				ctx, cancel = context.WithCancel(context.Background())
			}
			n := 0
			resp, err := llm.GenerateContent(ctx, msgs, llms.WithStreamingFunc(func(_ context.Context, c streaming.Chunk) error {
				if c.Type == streaming.ChunkTypeText {
					n++
					if n == 3 && mode != "deadline" {
						cancel()
					}
				}
				return nil
			}))
			cancel()
			if err == nil {
				nilErr++
				if !shown {
					shown = true
					fmt.Printf("[%s] example nil-error result: content=%q stop=%q\n", mode, resp.Choices[0].Content, resp.Choices[0].StopReason)
				}
			} else {
				withErr++
			}
		}
		fmt.Printf("[%s] nil error %d/30, error %d/30\n", mode, nilErr, withErr)
	}
}
