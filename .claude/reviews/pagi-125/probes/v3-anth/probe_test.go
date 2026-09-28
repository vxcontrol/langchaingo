package anthropic_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
)

func TestProbeV3Prefill(t *testing.T) {
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"This model does not support assistant message prefill. The conversation must end with a user message."}}`)
	}))
	defer srv.Close()
	for _, m := range []string{"claude-opus-4-7", "claude-opus-latest", "claude-sonnet-latest", "claude-fable-latest"} {
		llm, err := anthropic.New(anthropic.WithToken("x"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel(m))
		if err != nil {
			t.Fatal(err)
		}
		before := hits
		_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
			llms.TextParts(llms.ChatMessageTypeAI, "Sure, here is"),
		})
		fmt.Printf("model=%s sentToVendor=%v err=%v\n", m, hits > before, err)
	}
}
