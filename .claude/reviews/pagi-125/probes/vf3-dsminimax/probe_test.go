package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

type vf3rt struct{ target *url.URL }

func (r vf3rt) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = r.target.Scheme
	req.URL.Host = r.target.Host
	return http.DefaultTransport.RoundTrip(req)
}

func TestProbeVF3(t *testing.T) {
	for _, tc := range []struct{ base, model string }{
		{"https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "MiniMax-M2.5"},
		{"https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "MiniMax/MiniMax-M3"},
		{"https://gateway.example.com/v1", "dashscope/MiniMax-M2.5"},
		{"https://gateway.example.com/v1", "dashscope/MiniMax/MiniMax-M3"},
		{"https://gateway.example.com/v1", "minimax/MiniMax-M2.5"},
	} {
		var raw []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
		}))
		u, _ := url.Parse(srv.URL)
		llm, err := New(WithBaseURL(tc.base), WithToken("t"), WithModel(tc.model), WithHTTPClient(&http.Client{Transport: vf3rt{u}}))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithReasoning("", 4000))
		var warn any
		if resp != nil && len(resp.Choices) > 0 {
			warn = resp.Choices[0].GenerationInfo
		}
		t.Logf("%s @ %s: err=%v body=%s info=%v", tc.model, tc.base, err, raw, warn)
		srv.Close()
	}
}
