package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type v3Doer struct{ body []byte }

func (d *v3Doer) Do(r *http.Request) (*http.Response, error) {
	d.body, _ = io.ReadAll(r.Body)
	resp := `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"{\"a\":1}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(bytes.NewBufferString(resp)), Request: r}, nil
}

func v3wire(b []byte) string {
	if b == nil {
		return "<no request>"
	}
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	keep := map[string]any{}
	for _, k := range []string{"reasoning_effort", "reasoning", "temperature", "top_k", "response_format"} {
		if v, ok := m[k]; ok {
			if k == "response_format" {
				v = v.(map[string]any)["type"]
			}
			keep[k] = v
		}
	}
	if msgs, ok := m["messages"].([]any); ok {
		for _, mm := range msgs {
			mmm := mm.(map[string]any)
			if mmm["role"] == "assistant" {
				keep["asst.content"] = mmm["content"]
				if rc, ok := mmm["reasoning_content"]; ok {
					keep["asst.reasoning_content"] = rc
				}
			}
		}
	}
	ks := make([]string, 0, len(keep))
	for k := range keep {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	var sb strings.Builder
	for _, k := range ks {
		fmt.Fprintf(&sb, "%s=%v ", k, keep[k])
	}
	return sb.String()
}

func TestProbeV3OR(t *testing.T) {
	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{Name: "f", Description: "d", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}
	so := llms.StructuredOutputConfig{Name: "x", Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}`)}
	user := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
	withReasoning := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextContent{Text: "hello", Reasoning: &reasoning.ContentReasoning{Content: "THOUGHT"}}}},
		llms.TextParts(llms.ChatMessageTypeHuman, "again"),
	}
	cases := []struct {
		name, model string
		preserve    bool
		msgs        []llms.MessageContent
		opts        []llms.CallOption
	}{
		{"A1 gpt-5.4 tools+high", "openai/gpt-5.4", false, user, []llms.CallOption{llms.WithTools([]llms.Tool{tool}), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"A2 gpt-5.5 tools+high", "openai/gpt-5.5", false, user, []llms.CallOption{llms.WithTools([]llms.Tool{tool}), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"A3 gpt-5.6 tools only", "openai/gpt-5.6", false, user, []llms.CallOption{llms.WithTools([]llms.Tool{tool})}},
		{"A4 gpt-5.6 tools+high", "openai/gpt-5.6", false, user, []llms.CallOption{llms.WithTools([]llms.Tool{tool}), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"A5 gpt-5.4 high no tools", "openai/gpt-5.4", false, user, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"B1 minimax-m2 structured", "minimax/minimax-m2", false, user, []llms.CallOption{llms.WithStructuredOutput(so)}},
		{"B2 minimax-m2 topk", "minimax/minimax-m2", false, user, []llms.CallOption{llms.WithTopK(40)}},
		{"B3 minimax-m2 preserve", "minimax/minimax-m2", true, withReasoning, nil},
		{"B4 minimax-01 structured", "minimax/minimax-01", false, user, []llms.CallOption{llms.WithStructuredOutput(so)}},
		{"D1 claude-sonnet-4", "anthropic/claude-sonnet-4", false, user, []llms.CallOption{llms.WithTemperature(0.2), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"D2 claude-sonnet-4.5", "anthropic/claude-sonnet-4.5", false, user, []llms.CallOption{llms.WithTemperature(0.2), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"D3 claude-opus-4", "anthropic/claude-opus-4", false, user, []llms.CallOption{llms.WithTemperature(0.2), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"D4 claude-opus-4.1", "anthropic/claude-opus-4.1", false, user, []llms.CallOption{llms.WithTemperature(0.2), llms.WithReasoning(llms.ReasoningHigh, 0)}},
		{"D5 claude-sonnet-4 bare@anthropic host", "claude-sonnet-4-20250514", false, user, []llms.CallOption{llms.WithTemperature(0.2), llms.WithReasoning(llms.ReasoningHigh, 0)}},
	}
	for _, host := range []string{"https://openrouter.ai/api/v1"} {
		for _, c := range cases {
			d := &v3Doer{}
			o := []openai.Option{openai.WithToken("k"), openai.WithBaseURL(host), openai.WithModel(c.model), openai.WithHTTPClient(d)}
			if c.preserve {
				o = append(o, openai.WithPreserveReasoningContent())
			}
			llm, err := openai.New(o...)
			if err != nil {
				t.Fatal(err)
			}
			_, err = llm.GenerateContent(context.Background(), c.msgs, c.opts...)
			fmt.Printf("%s | err=%v | wire=%s\n", c.name, err, v3wire(d.body))
		}
	}
}
