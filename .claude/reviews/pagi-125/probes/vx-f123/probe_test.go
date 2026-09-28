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

type vxRT struct{ target *url.URL }

func (r vxRT) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.URL.Scheme = r.target.Scheme
	req.URL.Host = r.target.Host
	return http.DefaultTransport.RoundTrip(req)
}

func vxRun(t *testing.T, base, name, model string, extra []openai.Option, opts ...llms.CallOption) {
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
	o := append([]openai.Option{openai.WithBaseURL(base), openai.WithToken("x"), openai.WithModel(model), openai.WithHTTPClient(&http.Client{Transport: vxRT{target}})}, extra...)
	llm, err := openai.New(o...)
	if err != nil {
		t.Fatal(err)
	}
	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	var p map[string]any
	_ = json.Unmarshal(body, &p)
	delete(p, "messages")
	b, _ := json.Marshal(p)
	t.Logf("%s: err=%v body=%s", name, err, b)
}

func TestProbeVXF1(t *testing.T) {
	or := "https://openrouter.ai/api/v1"
	modern := []openai.Option{openai.WithModernReasoningFormat()}
	for _, m := range []string{"z-ai/glm-4.6", "qwen/qwen3-235b-a22b", "moonshotai/kimi-k2-thinking", "minimax/minimax-m2", "deepseek/deepseek-v4-pro", "openai/gpt-5", "deepseek/deepseek-r1"} {
		vxRun(t, or, "off "+m, m, modern, llms.WithReasoningDisabled())
		vxRun(t, or, "high "+m, m, modern, llms.WithReasoning(llms.ReasoningHigh, 0))
	}
}

func TestProbeVXF3(t *testing.T) {
	fn := []llms.FunctionDefinition{{Name: "f", Description: "d", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}
	tl := []llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{Name: "f", Description: "d", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}}
	oa := "https://api.openai.com/v1"
	for _, m := range []string{"gpt-5.6", "gpt-5.4"} {
		vxRun(t, oa, m+" functions default", m, nil, llms.WithFunctions(fn))
		vxRun(t, oa, m+" tools default", m, nil, llms.WithTools(tl))
		vxRun(t, oa, m+" functions high", m, nil, llms.WithFunctions(fn), llms.WithReasoning(llms.ReasoningHigh, 0))
		vxRun(t, oa, m+" tools high", m, nil, llms.WithTools(tl), llms.WithReasoning(llms.ReasoningHigh, 0))
	}
}
