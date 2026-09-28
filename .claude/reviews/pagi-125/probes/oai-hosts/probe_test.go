package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

var (
	probeOnce sync.Once
	probeMu   sync.Mutex
	probeSrv  *httptest.Server
	probeBody []byte
)

func probeCapture2(t *testing.T, model string, extra []openai.Option, callOpts ...llms.CallOption) (map[string]any, *llms.ContentResponse, error) {
	t.Helper()
	probeMu.Lock()
	defer probeMu.Unlock()
	probeOnce.Do(func() {
		probeSrv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			probeBody, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"{\"answer\":\"ok\"}"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		}))
	})
	srv := probeSrv
	probeBody = nil
	opts := append([]openai.Option{openai.WithBaseURL(srv.URL), openai.WithToken("x"), openai.WithModel(model)}, extra...)
	llm, err := openai.New(opts...)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, callOpts...)
	var payload map[string]any
	_ = json.Unmarshal(probeBody, &payload)
	delete(payload, "messages")
	return payload, resp, err
}

func TestProbeHosts(t *testing.T) {
	type c struct {
		name  string
		model string
		extra []openai.Option
		opts  []llms.CallOption
	}
	cases := []c{
		{"minimax-openrouter-jsonmode", "minimax/minimax-m2", nil, []llms.CallOption{llms.WithJSONMode()}},
		{"minimax-openrouter-topk", "minimax/minimax-m2", nil, []llms.CallOption{llms.WithTopK(20)}},
		{"qwen-openrouter-modern-effort", "qwen/qwen3-235b-a22b", []openai.Option{openai.WithModernReasoningFormat()}, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"qwen-openrouter-legacy-effort", "qwen/qwen3-235b-a22b", nil, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"glm46-openrouter-modern-effort", "z-ai/glm-4.6", []openai.Option{openai.WithModernReasoningFormat()}, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"kimi-openrouter-modern-effort", "moonshotai/kimi-k2-thinking", []openai.Option{openai.WithModernReasoningFormat()}, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"minimax-openrouter-modern-effort", "minimax/minimax-m2", []openai.Option{openai.WithModernReasoningFormat()}, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}},
	}
	for _, tc := range cases {
		p, resp, err := probeCapture2(t, tc.model, tc.extra, tc.opts...)
		b, _ := json.Marshal(p)
		var w any
		if resp != nil {
			w = resp.Warnings
		}
		t.Logf("%s: err=%v body=%s warnings=%+v", tc.name, err, b, w)
	}
}
