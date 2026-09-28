package openai_test

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
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func r2Send(t *testing.T, baseURL, model string, opts ...llms.CallOption) (string, error) {
	t.Helper()
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"{\"a\":1}"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	addr := srv.Listener.Addr().String()
	transport := &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, addr)
	}}
	defer transport.CloseIdleConnections()
	llm, err := openai.New(openai.WithBaseURL(baseURL), openai.WithToken("token"), openai.WithModel(model),
		openai.WithHTTPClient(&http.Client{Transport: transport}))
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	return body, err
}

func r2Fields(body string, keys ...string) string {
	if body == "" {
		return "<no request>"
	}
	var m map[string]any
	_ = json.Unmarshal([]byte(body), &m)
	out := map[string]any{}
	for _, k := range keys {
		if v, ok := m[k]; ok {
			out[k] = v
		}
	}
	b, _ := json.Marshal(out)
	return string(b)
}

func TestProbeR2Door(t *testing.T) {
	const or = "http://openrouter.ai/api/v1"
	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{Name: "get_weather", Description: "w",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}}}}
	keys := []string{"reasoning_effort", "reasoning", "temperature", "top_p", "top_k", "response_format", "thinking", "enable_thinking"}
	cases := []struct {
		name, base, model string
		opts              []llms.CallOption
	}{
		{"A1 OR gpt-5.4 tools+high", or, "openai/gpt-5.4", []llms.CallOption{llms.WithTools([]llms.Tool{tool}), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"A2 OR gpt-5.5 tools+high", or, "openai/gpt-5.5", []llms.CallOption{llms.WithTools([]llms.Tool{tool}), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"A3 OR gpt-5.6 tools no reasoning", or, "openai/gpt-5.6", []llms.CallOption{llms.WithTools([]llms.Tool{tool})}},
		{"A4 OR gpt-5.6 tools+high", or, "openai/gpt-5.6", []llms.CallOption{llms.WithTools([]llms.Tool{tool}), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"B1 OR minimax-m2 structured", or, "minimax/minimax-m2", []llms.CallOption{llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "x", Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}`)})}},
		{"B2 OR minimax-m2 topk", or, "minimax/minimax-m2", []llms.CallOption{llms.WithTopK(40)}},
		{"B3 OR deepseek-v4-flash temp+topp", or, "deepseek/deepseek-v4-flash", []llms.CallOption{llms.WithTemperature(0.3), llms.WithTopP(0.9), llms.WithTopK(20)}},
		{"C1 z.ai glm-latest medium", "http://api.z.ai/api/paas/v4", "glm-latest", []llms.CallOption{llms.WithReasoning(llms.ReasoningMedium, 0)}},
		{"C2 z.ai glm-5.3 medium", "http://api.z.ai/api/paas/v4", "glm-5.3", []llms.CallOption{llms.WithReasoning(llms.ReasoningMedium, 0)}},
		{"C3 z.ai glm-flash-latest xhigh", "http://api.z.ai/api/paas/v4", "glm-flash-latest", []llms.CallOption{llms.WithReasoning(llms.ReasoningXHigh, 0)}},
		{"C4 z.ai glm-5.3-flash xhigh", "http://api.z.ai/api/paas/v4", "glm-5.3-flash", []llms.CallOption{llms.WithReasoning(llms.ReasoningXHigh, 0)}},
		{"D1 OR claude-sonnet-4 temp+high", or, "anthropic/claude-sonnet-4", []llms.CallOption{llms.WithTemperature(0.2), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"D2 OR claude-sonnet-4.5 temp+high", or, "anthropic/claude-sonnet-4.5", []llms.CallOption{llms.WithTemperature(0.2), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"D3 OR claude-opus-4 temp+high", or, "anthropic/claude-opus-4", []llms.CallOption{llms.WithTemperature(0.2), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"D4 OR claude-opus-4.1 temp+high", or, "anthropic/claude-opus-4.1", []llms.CallOption{llms.WithTemperature(0.2), llms.WithReasoning(llms.ReasoningHigh, 0)}},
	}
	for _, tc := range cases {
		body, err := r2Send(t, tc.base, tc.model, tc.opts...)
		fmt.Printf("PROBE %-36s err=%v\n      wire=%s\n", tc.name, err, r2Fields(body, keys...))
	}
}
