package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func probeServer(t *testing.T) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		t.Logf("  request body: %s", body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"{\"a\":\"b\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
}

func run(t *testing.T, srv *httptest.Server, model string, opts ...llms.CallOption) {
	llm, err := openai.New(openai.WithToken("test"), openai.WithBaseURL(srv.URL+"/v1"), openai.WithModel(model))
	if err != nil {
		t.Fatal(err)
	}
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	t.Logf("  RESULT model=%s err=%v", model, err)
}

func TestProbeV1StructuredHost(t *testing.T) {
	srv := probeServer(t)
	defer srv.Close()
	schema := json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)
	for _, m := range []string{"deepseek/deepseek-chat-v3.1", "z-ai/glm-4.6", "zai-org/GLM-4.5-Air", "deepseek-ai/DeepSeek-V3"} {
		run(t, srv, m, llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "r", Schema: schema}))
	}
}

func TestProbeV1QwenHost(t *testing.T) {
	srv := probeServer(t)
	defer srv.Close()
	t.Log("non-streaming WithReasoning on qwen3-8b:")
	run(t, srv, "qwen3-8b", llms.WithReasoning(llms.ReasoningHigh, 0))
	t.Log("plain call on qwen3-8b:")
	run(t, srv, "qwen3-8b")
}
