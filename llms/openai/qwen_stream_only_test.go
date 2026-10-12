package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func captureQwenRequest(t *testing.T, model string, opts ...llms.CallOption) (map[string]any, error) {
	t.Helper()

	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		if stream, _ := body["stream"].(bool); stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\","+
				"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n"+
				"data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer server.Close()

	llm, err := New(WithBaseURL(server.URL), WithToken("token"), WithModel(model))
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	_, genErr := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	return body, genErr
}

func TestQwenHybridDisablesThinkingWithoutAStream(t *testing.T) {
	t.Parallel()

	body, err := captureQwenRequest(t, "dashscope/qwen3-8b")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got, ok := body["enable_thinking"]; !ok || got != false {
		t.Fatalf("a non-streaming call must carry enable_thinking=false, got %v (present=%v)", got, ok)
	}
}

func TestQwenHybridLeavesAStreamingCallAlone(t *testing.T) {
	t.Parallel()

	body, err := captureQwenRequest(t, "dashscope/qwen3-8b",
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got, ok := body["enable_thinking"]; ok {
		t.Fatalf("a streaming call must not pin thinking, got enable_thinking=%v", got)
	}
}

func TestQwenHybridRefusesThinkingWithoutAStream(t *testing.T) {
	t.Parallel()

	_, err := captureQwenRequest(t, "dashscope/qwen3-8b", llms.WithReasoning(llms.ReasoningHigh, 0))
	var want *reasoning.ErrThinkingRequiresStream
	if !errors.As(err, &want) {
		t.Fatalf("asking a stream-only model to think without a stream must be refused, got %v", err)
	}
}

func TestQwenCoderKeepsTheBareCallUntouched(t *testing.T) {
	t.Parallel()

	body, err := captureQwenRequest(t, "dashscope/qwen3-coder-plus")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got, ok := body["enable_thinking"]; ok {
		t.Fatalf("the coder line is not a hybrid, got enable_thinking=%v", got)
	}
}

func TestQwenHybridHonoursDisabledThinkingOnAStream(t *testing.T) {
	t.Parallel()

	body, err := captureQwenRequest(t, "dashscope/qwen3-8b",
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }),
		llms.WithReasoningDisabled())
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got, ok := body["enable_thinking"]; !ok || got != false {
		t.Fatalf("an explicit disable must reach the wire, got %v (present=%v)", got, ok)
	}
}

func TestQwenHybridKeepsTheCallerSampling(t *testing.T) {
	t.Parallel()

	body, err := captureQwenRequest(t, "dashscope/qwen3-8b",
		llms.WithTemperature(0.25), llms.WithTopP(0.3))
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got := body["temperature"]; got != 0.25 {
		t.Fatalf("the caller's temperature must survive, got %v", got)
	}
	if got := body["top_p"]; got != 0.3 {
		t.Fatalf("the caller's top_p must survive, got %v", got)
	}
}

func TestQwenCommercialHybridTurnsThinkingOnWhenAsked(t *testing.T) {
	t.Parallel()

	body, err := captureQwenRequest(t, "dashscope/qwen-plus", llms.WithReasoning(llms.ReasoningHigh, 0))
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got, ok := body["enable_thinking"]; !ok || got != true {
		t.Fatalf("asking for reasoning must turn the vendor flag on, got %v (present=%v)", got, ok)
	}
}

func TestQwenCommercialHybridStaysQuietOtherwise(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		opts []llms.CallOption
	}{
		{"bare", nil},
		{"disabled", []llms.CallOption{llms.WithReasoningDisabled()}},
	} {
		body, err := captureQwenRequest(t, "dashscope/qwen-plus", tc.opts...)
		if err != nil {
			t.Fatalf("%s: generate: %v", tc.name, err)
		}
		if got, ok := body["enable_thinking"]; ok {
			t.Errorf("%s: the flag must stay off the wire, got %v", tc.name, got)
		}
	}
}

func TestModelStudioServesOpenWeightQwensThinkingWithoutAStream(t *testing.T) {
	t.Parallel()

	var streamOnly *reasoning.ErrThinkingRequiresStream
	for _, at := range []struct{ baseURL, prefix string }{
		{modelStudioBaseURL, ""},
		{privateGatewayBaseURL, "dashscope/"},
	} {
		for _, model := range []string{
			"qwen3.5-27b", "qwen3.5-122b-a10b", "qwen3.6-27b", "qwen3.6-35b-a3b", "qwen3.8-27b", "qwen3.8-2.4t-a95b",
		} {
			name := at.prefix + model
			for _, think := range [][]llms.CallOption{
				nil, {llms.WithReasoning(llms.ReasoningHigh, 0)}, {llms.WithExtraBody(map[string]any{"thinking_budget": 4096})},
			} {
				body, err := callWithoutModel(t, at.baseURL, name, think...)
				if errors.As(err, &streamOnly) || body == nil || body["enable_thinking"] == false || body["reasoning_effort"] == "none" {
					t.Errorf("%s: Model Studio serves the model's thinking without a stream, got %v, %v", name, body, err)
				}
			}
		}
	}
}

func TestQwen38IsSwitchedOffByEffortNone(t *testing.T) {
	t.Parallel()

	for _, at := range []struct{ baseURL, prefix string }{
		{modelStudioBaseURL, ""},
		{privateGatewayBaseURL, "dashscope/"},
	} {
		for _, model := range []string{"qwen3.8-27b", "qwen3.8-max", "qwen3.8-flash", "qwen3.8-omni-flash"} {
			name := at.prefix + model
			body, err := callWithoutModel(t, at.baseURL, name, llms.WithReasoningDisabled())
			if err != nil || body["reasoning_effort"] != "none" {
				t.Errorf("%s: Off must go out as reasoning_effort none, got %v, %v", name, body, err)
			}
			if _, sent := body["enable_thinking"]; sent {
				t.Errorf("%s: Off goes out as reasoning_effort alone, got %v", name, body)
			}
		}
	}
}

func TestAStreamOnlyQwenRefusesARequestToThinkWithoutAStream(t *testing.T) {
	t.Parallel()

	var streamOnly *reasoning.ErrThinkingRequiresStream
	for _, at := range []struct{ baseURL, model string }{
		{modelStudioBaseURL, "qwen3-32b"},
		{privateGatewayBaseURL, "dashscope/qwen3-32b"},
	} {
		for _, think := range []llms.CallOption{
			llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithExtraBody(map[string]any{"enable_thinking": true}),
			llms.WithExtraBody(map[string]any{"thinking_budget": 8000}), llms.WithExtraBody(map[string]any{"thinking_budget": 0}),
		} {
			body, err := callWithoutModel(t, at.baseURL, at.model, think)
			if !errors.As(err, &streamOnly) || body != nil {
				t.Errorf("%s: thinking without a stream must be refused before the call, got %v", at.model, err)
			}
		}

		for _, off := range []llms.CallOption{llms.WithMaxTokens(512), llms.WithReasoningDisabled()} {
			body, err := callWithoutModel(t, at.baseURL, at.model, off,
				llms.WithExtraBody(map[string]any{"enable_thinking": false, "thinking_budget": 2048}))
			if err != nil || body["enable_thinking"] != false || body["thinking_budget"] != 2048.0 {
				t.Errorf("%s: an extra body that switches thinking off itself must go out, got %v, %v", at.model, body, err)
			}
		}

		body, err := callWithoutModel(t, at.baseURL, at.model, llms.WithReasoningDisabled(),
			llms.WithExtraBody(map[string]any{"thinking_budget": 4096}))
		if err != nil || body["enable_thinking"] != false || body["thinking_budget"] != 4096.0 {
			t.Errorf("%s: a caller that switches thinking off is not asking to think by a budget, got %v, %v", at.model, body, err)
		}

		if _, err := callWithoutModel(t, at.baseURL, at.model, llms.WithExtraBody(map[string]any{"thinking_budget": nil})); err != nil {
			t.Errorf("%s: a null budget asks for nothing, got %v", at.model, err)
		}
	}
}
