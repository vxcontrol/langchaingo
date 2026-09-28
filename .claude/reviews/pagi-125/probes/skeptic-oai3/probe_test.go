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

type skRT struct{ u *url.URL }

func (r skRT) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = r.u.Scheme
	req.URL.Host = r.u.Host
	return http.DefaultTransport.RoundTrip(req)
}

func TestProbeSkeptic3(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"logprobs":{"content":[{"token":"ok","logprob":-0.1,"top_logprobs":[{"token":"ok","logprob":-0.1}]}]},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	target, _ := url.Parse(srv.URL)
	hc := &http.Client{Transport: skRT{target}}
	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{Name: "f", Description: "d", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}
	fn := llms.FunctionDefinition{Name: "f", Description: "d", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}
	run := func(name, base, model string, extra []openai.Option, opts ...llms.CallOption) {
		o := append([]openai.Option{openai.WithBaseURL(base), openai.WithToken("x"), openai.WithModel(model), openai.WithHTTPClient(hc)}, extra...)
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
		var gi any
		if resp != nil && len(resp.Choices) > 0 {
			c, _ := json.Marshal(resp.Choices[0])
			gi = string(c)
		}
		t.Logf("%s: err=%v body=%s choice=%v", name, err, b, gi)
	}
	or := "https://openrouter.ai/api/v1"
	oa := "https://api.openai.com/v1"
	m := []openai.Option{openai.WithModernReasoningFormat()}
	for _, model := range []string{"z-ai/glm-4.6", "qwen/qwen3-235b-a22b", "moonshotai/kimi-k2-thinking", "minimax/minimax-m2", "deepseek/deepseek-v4-pro", "openai/gpt-5"} {
		run("F1 off "+model, or, model, m, llms.WithReasoningDisabled())
		run("F1 high "+model, or, model, m, llms.WithReasoning(llms.ReasoningHigh, 0))
	}
	run("F2 lp+top", oa, "gpt-4o", nil, llms.WithLogProbs(true), llms.WithTopLogProbs(1))
	run("F2 top only", oa, "gpt-4o", nil, llms.WithTopLogProbs(3))
	for _, model := range []string{"gpt-5.6", "gpt-5.4"} {
		run("F3 functions default "+model, oa, model, nil, llms.WithFunctions([]llms.FunctionDefinition{fn}))
		run("F3 tools default "+model, oa, model, nil, llms.WithTools([]llms.Tool{tool}))
		run("F3 functions high "+model, oa, model, nil, llms.WithFunctions([]llms.FunctionDefinition{fn}), llms.WithReasoning(llms.ReasoningHigh, 0))
		run("F3 tools high "+model, oa, model, nil, llms.WithTools([]llms.Tool{tool}), llms.WithReasoning(llms.ReasoningHigh, 0))
	}
}
