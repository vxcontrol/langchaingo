package openai

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeVF1(t *testing.T) {
	for _, tc := range []struct{ base, model string }{
		{"http://dashscope-intl.aliyuncs.com/compatible-mode/v1", "MiniMax-M2.5"},
		{"http://litellm.example/v1", "dashscope/MiniMax-M2.5"},
		{"http://dashscope-intl.aliyuncs.com/compatible-mode/v1", "Moonshot-Kimi-K2-Instruct"},
		{"http://litellm.example/v1", "dashscope/Moonshot-Kimi-K2-Instruct"},
		{"http://dashscope-intl.aliyuncs.com/compatible-mode/v1", "glm-4.5-air"},
		{"http://litellm.example/v1", "dashscope/glm-4.5-air"},
	} {
		var body map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		}))
		addr := srv.Listener.Addr().String()
		tr := &http.Transport{DialContext: func(ctx context.Context, n, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, n, addr)
		}}
		llm, err := New(WithBaseURL(tc.base), WithToken("t"), WithModel(tc.model), WithHTTPClient(&http.Client{Transport: tr}))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithReasoning("", 4000))
		var warn any
		if resp != nil && len(resp.Choices) > 0 {
			warn = resp.Choices[0].GenerationInfo["warnings"]
		}
		sent := body != nil
		t.Logf("%-60s %-38s err=%v sent=%v warn=%v body.reasoning=%v/%v/%v", tc.base, tc.model, err, sent, warn, body["reasoning_effort"], body["thinking_budget"], body["thinking"])
		srv.Close()
	}
}
