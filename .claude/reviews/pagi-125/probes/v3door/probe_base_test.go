package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func v3send(t *testing.T, baseURL, model string, opts ...llms.CallOption) (string, error) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	addr := srv.Listener.Addr().String()
	tr := &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, addr)
	}}
	llm, err := New(WithBaseURL(baseURL), WithToken("token"), WithModel(model), WithHTTPClient(&http.Client{Transport: tr}))
	if err != nil {
		return "", err
	}
	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	out := map[string]any{}
	for _, k := range []string{"reasoning_effort", "temperature", "top_p", "top_k", "thinking"} {
		if v, ok := body[k]; ok {
			out[k] = v
		}
	}
	b, _ := json.Marshal(out)
	return string(b), err
}

func TestProbeV3Door(t *testing.T) {
	zai := "http://api.z.ai/api/paas/v4"
	for _, c := range []struct {
		m string
		e llms.ReasoningEffort
	}{{"glm-latest", llms.ReasoningMedium}, {"glm-5.3", llms.ReasoningMedium}, {"glm-flash-latest", llms.ReasoningXHigh}, {"glm-5.3-flash", llms.ReasoningXHigh}, {"glm-latest", llms.ReasoningLow}, {"glm-5.3", llms.ReasoningLow}} {
		w, err := v3send(t, zai, c.m, llms.WithReasoning(c.e, 0))
		fmt.Printf("ZAI %s %s wire=%s err=%v\n", c.m, c.e, w, err)
	}
	or := "http://openrouter.ai/api/v1"
	for _, m := range []string{"deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro", "deepseek-ai/deepseek-v4-flash"} {
		w, err := v3send(t, or, m, llms.WithTemperature(0.3), llms.WithTopP(0.9), llms.WithTopK(20))
		fmt.Printf("OR %s plain wire=%s err=%v\n", m, w, err)
		w, err = v3send(t, or, m, llms.WithTemperature(0.3), llms.WithTopP(0.9), llms.WithReasoning(llms.ReasoningHigh, 0))
		fmt.Printf("OR %s reasoning wire=%s err=%v\n", m, w, err)
	}
}
