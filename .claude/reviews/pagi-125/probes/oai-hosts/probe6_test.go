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

func TestProbeDashScopeRawVsRoute(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	target, _ := url.Parse(srv.URL)
	hc := &http.Client{Transport: redirectRT{target}}
	run := func(name, base, model string, opts ...llms.CallOption) {
		llm, err := openai.New(openai.WithBaseURL(base), openai.WithToken("x"), openai.WithModel(model), openai.WithHTTPClient(hc))
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
			w = resp.Warnings
		}
		t.Logf("%s: err=%v body=%s warn=%v", name, err, b, w)
	}
	ds := "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	gw := "https://gateway.example.com/v1"
	for _, m := range []string{"MiniMax-M2.5", "kimi-k2-thinking", "glm-4.6"} {
		run(m+" on dashscope host budget", ds, m, llms.WithReasoning("", 4000))
		run("dashscope/"+m+" on gateway budget", gw, "dashscope/"+m, llms.WithReasoning("", 4000))
		run(m+" on dashscope host off", ds, m, llms.WithReasoningDisabled())
		run("dashscope/"+m+" on gateway off", gw, "dashscope/"+m, llms.WithReasoningDisabled())
	}
}
