package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func weatherTool() llms.Tool {
	return llms.Tool{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:       "get_weather",
			Parameters: map[string]any{"type": "object"},
		},
	}
}

func bodyForCall(t *testing.T, model string, opts ...llms.CallOption) string {
	t.Helper()

	const completion = `{"id":"x","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completion)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithBaseURL(srv.URL), WithToken("test"), WithModel(model))
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if _, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...); err != nil {
		t.Fatalf("GenerateContent() error: %v", err)
	}
	return body
}

func TestEffortAndToolsOnTheWire(t *testing.T) {
	t.Parallel()

	tools := llms.WithTools([]llms.Tool{weatherTool()})
	high := llms.WithReasoning(llms.ReasoningHigh, 0)

	for _, tc := range []struct {
		name   string
		model  string
		opts   []llms.CallOption
		want   string
		absent bool
	}{
		{"5.6 with tools and a level degrades to none", "gpt-5.6-sol",
			[]llms.CallOption{tools, high}, `"reasoning_effort":"none"`, false},
		{"5.6 with tools and no request still sends none", "gpt-5.6-sol",
			[]llms.CallOption{tools}, `"reasoning_effort":"none"`, false},
		{"5.6 without tools keeps the level", "gpt-5.6-sol",
			[]llms.CallOption{high}, `"reasoning_effort":"high"`, false},
		{"5.5 with tools omits the field", "gpt-5.5",
			[]llms.CallOption{tools, high}, `"reasoning_effort"`, true},
		{"5.4-nano with tools omits the field", "gpt-5.4-nano",
			[]llms.CallOption{tools, high}, `"reasoning_effort"`, true},
		{"5.5 without tools keeps the level", "gpt-5.5",
			[]llms.CallOption{high}, `"reasoning_effort":"high"`, false},
		{"5.2 with tools keeps the level", "gpt-5.2",
			[]llms.CallOption{tools, high}, `"reasoning_effort":"high"`, false},
		{"gpt-4o never gets the field", "gpt-4o",
			[]llms.CallOption{high}, `"reasoning_effort"`, true},
		{"gpt-4o with tools also omits it", "gpt-4o",
			[]llms.CallOption{tools, high}, `"reasoning_effort"`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body := bodyForCall(t, tc.model, tc.opts...)
			if got := strings.Contains(body, tc.want); got == tc.absent {
				verb := "does not contain"
				if tc.absent {
					verb = "contains"
				}
				t.Errorf("request body %s %s\nbody: %s", verb, tc.want, body)
			}
		})
	}
}

func TestDashScopeThinkingBudgetOnTheWire(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name   string
		model  string
		opt    llms.CallOption
		want   string
		absent bool
	}{
		{"budget is sent instead of effort", "dashscope/qwen3.8-max",
			llms.WithReasoning(llms.ReasoningLow, 4096), `"thinking_budget":4096`, false},
		{"effort is omitted when a budget is set", "dashscope/qwen3.8-max",
			llms.WithReasoning(llms.ReasoningLow, 4096), `"reasoning_effort"`, true},
		{"without a budget, effort is sent", "dashscope/qwen3.8-max",
			llms.WithReasoning(llms.ReasoningLow, 0), `"reasoning_effort":"low"`, false},
		{"without a budget, the budget field is absent", "dashscope/qwen3.8-max",
			llms.WithReasoning(llms.ReasoningLow, 0), `"thinking_budget"`, true},
		{"3.7 also takes a budget", "dashscope/qwen3.7-plus",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"qwen-plus takes a budget", "dashscope/qwen-plus",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"qwen-flash takes a budget", "dashscope/qwen-flash",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"qwen3-max takes a budget", "dashscope/qwen3-max",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"qwen3-vl-plus takes a budget", "dashscope/qwen3-vl-plus",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"qwen3-vl-flash takes a budget", "dashscope/qwen3-vl-flash",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"qwen3-next takes a budget", "dashscope/qwen3-next-80b-a3b-thinking",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"qwen-turbo takes a budget", "dashscope/qwen-turbo",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"open-weight VL takes a budget", "dashscope/qwen3-vl-30b-a3b",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"large open-weight VL does too", "dashscope/qwen3-vl-235b-a22b",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"non-thinking next does not get a budget", "dashscope/qwen3-next-80b-a3b-instruct",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
		{"the flag stays alongside the budget", "dashscope/qwen-plus",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"enable_thinking":true`, false},
		{"guest glm-5.2 takes a budget", "dashscope/glm-5.2",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"glm-5.2 generation snapshot does too", "dashscope/glm-5.2-fast-preview",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"guest glm-5.1 takes a budget", "dashscope/glm-5.1",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"guest kimi takes a budget", "dashscope/kimi-k2.7-code",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"guest deepseek-v4 takes a budget", "dashscope/deepseek-v4-pro",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"dated deepseek-v4 snapshot does too", "dashscope/deepseek-v4-flash-0731",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"guest omits effort when a budget is set", "dashscope/glm-5.2",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"reasoning_effort"`, true},
		{"deepseek-v3.2 does not get a budget", "dashscope/deepseek-v3.2",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
		{"guest without a gateway name does not get a budget", "glm-5.2",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
		{"bare kimi does not get a budget", "kimi-k2.7-code",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
		{"bare deepseek-v4 does not get a budget", "deepseek-v4-pro",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
		{"bare qwen still matches the gateway", "qwen-plus",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget":2048`, false},
		{"glm on its own gateway does not get a budget", "zai/glm-5.2",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
		{"kimi on its own gateway does not get a budget", "moonshot/kimi-k2.7-code",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
		{"glm via openrouter does not get a budget", "openrouter/z-ai/glm-5.2",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
		{"deepseek on its own gateway does not get a budget", "deepseek/deepseek-v4-pro",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
		{"a different host does not get a budget", "openrouter/qwen/qwen3.7-plus",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
		{"non-qwen does not get a budget", "gpt-5.2",
			llms.WithReasoning(llms.ReasoningLow, 2048), `"thinking_budget"`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body := bodyForCall(t, tc.model, llms.WithMaxTokens(8000), tc.opt)
			if got := strings.Contains(body, tc.want); got == tc.absent {
				verb := "does not contain"
				if tc.absent {
					verb = "contains"
				}
				t.Errorf("request body %s %s\nbody: %s", verb, tc.want, body)
			}
		})
	}
}
