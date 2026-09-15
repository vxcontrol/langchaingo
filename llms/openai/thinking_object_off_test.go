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
		"deepseek-v4-flash", "deepseek-v4-pro",
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
