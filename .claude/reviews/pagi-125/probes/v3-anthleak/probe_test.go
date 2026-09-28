package anthropic_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func v3Stuck() int {
	buf := make([]byte, 1<<20)
	n := runtime.Stack(buf, true)
	return strings.Count(string(buf[:n]), "parseStreamingCompletionResponse.func1")
}

func v3Run(t *testing.T, extraLines int) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		for i := 0; i <= extraLines; i++ {
			_, _ = fmt.Fprint(w, "event: completion\ndata: {\"completion\":\" Hello\",\"stop_reason\":null,\"model\":\"claude-2.1\"}\n\n")
		}
		fl.Flush()
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer srv.Close()
	llm, err := anthropic.New(anthropic.WithToken("k"), anthropic.WithBaseURL(srv.URL),
		anthropic.WithModel("claude-2.1"), anthropic.WithLegacyTextCompletionsAPI())
	if err != nil {
		t.Fatal(err)
	}
	before := v3Stuck()
	for i := 0; i < 5; i++ {
		_, err = llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return errors.New("consumer gave up") }))
	}
	time.Sleep(3 * time.Second)
	fmt.Printf("extraLines=%d err=%v leaked=%d\n", extraLines, err, v3Stuck()-before)
}

func TestProbeV3Leak(t *testing.T) {
	v3Run(t, 0)
	v3Run(t, 3)
}
