package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

type vxDoer struct{ body []byte }

func (d *vxDoer) Do(r *http.Request) (*http.Response, error) {
	d.body, _ = io.ReadAll(r.Body)
	resp := `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"{\"a\":\"b\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(bytes.NewBufferString(resp)), Request: r}, nil
}

func vxRun(t *testing.T, base, model string, extra []openai.Option, opts ...llms.CallOption) {
	t.Helper()
	d := &vxDoer{}
	o := []openai.Option{openai.WithToken("k"), openai.WithBaseURL(base), openai.WithModel(model), openai.WithHTTPClient(d)}
	o = append(o, extra...)
	llm, err := openai.New(o...)
	if err != nil {
		t.Fatal(err)
	}
	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	var m map[string]any
	_ = json.Unmarshal(d.body, &m)
	delete(m, "messages")
	delete(m, "tools")
	b, _ := json.Marshal(m)
	t.Logf("  %-40s err=%v body=%s", model, err, b)
}

func TestProbeVX1QwenOff(t *testing.T) {
	for _, h := range []struct {
		base string
		ex   []openai.Option
	}{{"http://vllm:8000/v1", nil}, {"https://openrouter.ai/api/v1", []openai.Option{openai.WithModernReasoningFormat()}}, {"https://api.together.xyz/v1", nil}} {
		t.Logf("host %s", h.base)
		for _, m := range []string{"Qwen/Qwen3-8B", "qwen/qwen3-32b", "Qwen/QwQ-32B", "qwq-32b", "Qwen/Qwen3-30B-A3B-Thinking-2507", "Qwen/Qwen3-235B-A22B-Instruct-2507"} {
			vxRun(t, h.base, m, h.ex, llms.WithReasoningDisabled())
			vxRun(t, h.base, m, h.ex, llms.WithReasoning(llms.ReasoningHigh, 0))
		}
	}
	t.Logf("dashscope")
	vxRun(t, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen3-32b", nil, llms.WithReasoningDisabled())
	vxRun(t, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen-plus", nil, llms.WithReasoningDisabled())
}

func TestProbeVX2ToolChoice(t *testing.T) {
	tools := []llms.Tool{
		{Type: "function", Function: &llms.FunctionDefinition{Name: "a", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}},
		{Type: "function", Function: &llms.FunctionDefinition{Name: "b", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}},
	}
	for name, c := range map[string]any{
		"map-any":    map[string]any{"type": "function", "function": map[string]any{"name": "b"}},
		"map-string": map[string]any{"type": "function", "function": map[string]string{"name": "b"}},
		"funcref":    map[string]any{"type": "function", "function": llms.FunctionReference{Name: "b"}},
		"funcref-p":  map[string]any{"type": "function", "function": &llms.FunctionReference{Name: "b"}},
		"struct":     llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "b"}},
	} {
		t.Logf("%s:", name)
		vxRun(t, "https://api.openai.com/v1", "gpt-4.1", nil, llms.WithTools(tools), llms.WithToolChoice(c))
	}
}

func TestProbeVX3OpenRouterSlugs(t *testing.T) {
	or := "https://openrouter.ai/api/v1"
	schema := llms.StructuredOutputConfig{Name: "r", Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)}
	for _, m := range []string{"minimax/minimax-m2", "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "deepseek/deepseek-chat-v3.1"} {
		t.Logf("model %s", m)
		vxRun(t, or, m, nil, llms.WithJSONMode())
		vxRun(t, or, m, nil, llms.WithStructuredOutput(schema))
		vxRun(t, or, m, nil, llms.WithTemperature(0.3), llms.WithTopK(40))
	}
	t.Logf("preserve reasoning minimax on openrouter")
	vxRun(t, or, "minimax/minimax-m2", []openai.Option{openai.WithPreserveReasoningContent()}, llms.WithTopK(40))
}
