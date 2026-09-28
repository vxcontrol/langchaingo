package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestDisablingThinkingWhereTheEffortTokenDoesNotReachTheVendor(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"glm-5.1", "glm-5.2", "glm-4.6", "minimax-m3", "kimi-k2.6",
		"deepseek-v4-flash", "deepseek-v4-pro", "deepseek-flash", "deepseek/deepseek-flash",
		"zai/glm-5.2", "deepseek/deepseek-v4-pro",
	} {
		body := sendForWire(t, model, llms.WithReasoningDisabled())
		if !strings.Contains(body, `"thinking":{"type":"disabled"}`) {
			t.Errorf("%s: body carries no disabled thinking object: %s", model, body)
		}
		if strings.Contains(body, "reasoning_effort") {
			t.Errorf("%s: body still carries reasoning_effort: %s", model, body)
		}
	}
}

func TestDisablingThinkingOnAHostThatServesTheFamilyUnderItsOwnName(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model string
		want  map[string]any
	}{
		{"dashscope/glm-5.2", map[string]any{"enable_thinking": false}},
		{"dashscope/deepseek-v4-pro", map[string]any{"enable_thinking": false}},
		{"dashscope/deepseek-v4-flash-0731", map[string]any{"enable_thinking": false}},
		{"mistral/zai-glm-5-2", map[string]any{}},
		{"zai-glm-5-2", map[string]any{}},
	} {
		body, err := wireBodyOf(t, tc.model, nil, llms.WithReasoningDisabled())
		if err != nil {
			t.Fatalf("%s: GenerateContent() error: %v", tc.model, err)
		}
		got := map[string]any{}
		for _, key := range []string{"thinking", "reasoning_effort", "reasoning", "enable_thinking", "thinking_budget"} {
			if value, ok := body[key]; ok {
				got[key] = value
			}
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: reasoning fields on the wire = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestDashScopeHostGetsTheGuestWireForNamesWithoutTheRoutePrefix(t *testing.T) {
	t.Parallel()

	const (
		intl      = "http://dashscope-intl.aliyuncs.com/compatible-mode/v1"
		hongKong  = "http://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1"
		workspace = "http://llm-abc.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
		zai       = "http://api.z.ai/api/paas/v4"
		moonshot  = "http://api.moonshot.ai/v1"
	)
	off := llms.WithReasoningDisabled()
	budget := llms.WithReasoning(llms.ReasoningNone, 2048)
	disabled := map[string]any{"enable_thinking": false}
	for _, tc := range []struct {
		baseURL, model string
		opt            llms.CallOption
		want           map[string]any
	}{
		{dashScopeBaseURL, "glm-5.2", off, disabled},
		{dashScopeBaseURL, "deepseek-v4-pro", off, disabled},
		{intl, "deepseek-v4-flash-0731", off, disabled},
		{hongKong, "kimi-k2.6", off, disabled},
		{workspace, "glm-5.1", off, disabled},
		{dashScopeBaseURL, "kimi/kimi-k2.6", off, disabled},
		{gatewayBaseURL, "dashscope/kimi/kimi-k2.6", off, disabled},
		{dashScopeBaseURL, "glm-5.2", budget, map[string]any{"thinking_budget": float64(2048)}},
		{workspace, "glm-5.1", budget, map[string]any{"thinking_budget": float64(2048)}},
		{dashScopeBaseURL, "kimi/kimi-k2.6", budget, map[string]any{"thinking_budget": float64(2048)}},
		{gatewayBaseURL, "dashscope/kimi/kimi-k2.6", budget, map[string]any{"thinking_budget": float64(2048)}},
		{dashScopeBaseURL, "kimi-k2.6", budget,
			map[string]any{"thinking_budget": float64(2048), "enable_thinking": true}},
		{dashScopeBaseURL, "glm-5.1", llms.WithReasoning(llms.ReasoningHigh, 0),
			map[string]any{"reasoning_effort": "high"}},
		{zai, "glm-5.2", off, map[string]any{"thinking": map[string]any{"type": "disabled"}}},
		{moonshot, "kimi-k2.6", llms.WithReasoning(llms.ReasoningHigh, 0), map[string]any{}},
	} {
		t.Run(tc.baseURL+" "+tc.model, func(t *testing.T) {
			body, _ := sendToHost(t, tc.baseURL, tc.model, tc.opt)
			got := map[string]any{}
			for _, key := range []string{"thinking", "reasoning_effort", "reasoning", "enable_thinking", "thinking_budget"} {
				if value, ok := body[key]; ok {
					got[key] = value
				}
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("reasoning fields on the wire = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDisablingThinkingOnAModelThatOnlyThinks(t *testing.T) {
	t.Parallel()

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

	llm, err := New(WithBaseURL(srv.URL), WithToken("test"), WithModel("qwen3.8-2.4t-a95b"))
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithReasoningDisabled())

	var unsupported *reasoning.ErrReasoningOffUnsupported
	if !errors.As(err, &unsupported) {
		t.Fatalf("want ErrReasoningOffUnsupported, got err=%v body=%s", err, body)
	}
	if body != "" {
		t.Errorf("the refusal must come before the network, but a request went out: %s", body)
	}
}
