package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func TestProbeOpenRouterHost(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"{\"answer\":\"ok\"}"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	target, _ := url.Parse(srv.URL)
	hc := &http.Client{Transport: redirectRT{target}}
	schema := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`)
	run := func(name, model string, extra []openai.Option, opts ...llms.CallOption) {
		o := append([]openai.Option{openai.WithBaseURL("https://openrouter.ai/api/v1"), openai.WithToken("x"), openai.WithModel(model), openai.WithHTTPClient(hc)}, extra...)
		llm, err := openai.New(o...)
		if err != nil {
			t.Fatal(err)
		}
		body = nil
		resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
		var p map[string]any
		_ = json.Unmarshal(body, &p)
		delete(p, "messages")
		b, _ := json.Marshal(p)
		var w any
		if resp != nil {
			w = warningsOf(resp)
		}
		t.Logf("%s: err=%v body=%s warn=%v", name, err, b, w)
	}
	run("minimax json", "minimax/minimax-m2", nil, llms.WithJSONMode())
	run("minimax schema", "minimax/minimax-m2", nil, llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "a", Schema: schema}))
	run("minimax topk", "minimax/minimax-m2", nil, llms.WithTopK(40))
	run("deepseek v4 temp", "deepseek/deepseek-v4-pro", nil, llms.WithTemperature(0.3), llms.WithTopP(0.9), llms.WithTopK(40))
	run("qwen off", "qwen/qwen3-32b", []openai.Option{openai.WithModernReasoningFormat()}, llms.WithReasoningDisabled())
	run("glm off", "z-ai/glm-4.6", []openai.Option{openai.WithModernReasoningFormat()}, llms.WithReasoningDisabled())
}
