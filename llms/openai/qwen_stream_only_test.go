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

func TestQwen35OpenWeightsThinkOnlyOnAStream(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"qwen3.5-122b-a10b", "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-397b-a17b"} {
		body, err := captureQwenRequest(t, "dashscope/"+model)
		if err != nil {
			t.Fatalf("%s: generate: %v", model, err)
		}
		if got, ok := body["enable_thinking"]; !ok || got != false {
			t.Errorf("%s: a non-streaming call must carry enable_thinking=false, got %v (present=%v)", model, got, ok)
		}

		_, err = captureQwenRequest(t, "dashscope/"+model, llms.WithReasoning(llms.ReasoningHigh, 0))
		var streamOnly *reasoning.ErrThinkingRequiresStream
		if !errors.As(err, &streamOnly) {
			t.Errorf("%s: thinking without a stream must be refused before the call, got %v", model, err)
		}

		body, err = captureQwenRequest(t, "dashscope/"+model,
			llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
		if err != nil {
			t.Fatalf("%s: generate: %v", model, err)
		}
		if got, ok := body["enable_thinking"]; ok {
			t.Errorf("%s: a streaming call must not pin thinking, got enable_thinking=%v", model, got)
		}
	}
}

func TestModelStudioServesEveryOpenWeightQwensThinkingOnlyOnAStream(t *testing.T) {
	t.Parallel()

	stream := llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil })
	var streamOnly *reasoning.ErrThinkingRequiresStream
	for _, at := range []struct{ baseURL, prefix string }{
		{modelStudioBaseURL, ""},
		{privateGatewayBaseURL, "dashscope/"},
	} {
		for model, off := range map[string][2]any{
			"qwen3.6-27b":     {false, nil},
			"qwen3.6-35b-a3b": {false, nil},
			"qwen3.8-27b":     {nil, "none"},
		} {
			name := at.prefix + model
			body, err := callWithoutModel(t, at.baseURL, name)
			if err != nil {
				t.Fatalf("%s: generate: %v", name, err)
			}
			if got := [2]any{body["enable_thinking"], body["reasoning_effort"]}; got != off {
				t.Errorf("%s: a call without a stream must switch thinking off with %v, got enable_thinking and reasoning_effort %v",
					name, off, got)
			}

			for _, think := range []llms.CallOption{
				llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithExtraBody(map[string]any{"enable_thinking": true}),
				llms.WithExtraBody(map[string]any{"thinking_budget": 8000}), llms.WithExtraBody(map[string]any{"thinking_budget": 0}),
			} {
				body, err = callWithoutModel(t, at.baseURL, name, think)
				if !errors.As(err, &streamOnly) || body != nil {
					t.Errorf("%s: thinking without a stream must be refused before the call, got %v", name, err)
				}
			}

			for _, off := range []llms.CallOption{llms.WithMaxTokens(512), llms.WithReasoningDisabled()} {
				body, err = callWithoutModel(t, at.baseURL, name, off,
					llms.WithExtraBody(map[string]any{"enable_thinking": false, "thinking_budget": 2048}))
				if err != nil || body["enable_thinking"] != false || body["reasoning_effort"] != nil || body["thinking_budget"] != 2048.0 {
					t.Errorf("%s: an extra body that switches thinking off itself must go out as written, got %v, %v", name, body, err)
				}
			}

			if _, err = callWithoutModel(t, at.baseURL, name, llms.WithExtraBody(map[string]any{"thinking_budget": nil})); err != nil {
				t.Errorf("%s: a null budget asks for nothing, got %v", name, err)
			}

			body, _ = callWithoutModel(t, at.baseURL, name, stream)
			if got := [2]any{body["enable_thinking"], body["reasoning_effort"]}; body["stream"] != true || got != [2]any{} {
				t.Errorf("%s: a streaming call must go out and leave the model's thinking alone, got %v", name, body)
			}
		}

		name := at.prefix + "qwen3.8-2.4t-a95b"
		for _, opts := range [][]llms.CallOption{nil, {llms.WithReasoningDisabled()}, {llms.WithReasoning(llms.ReasoningHigh, 0)}} {
			body, err := callWithoutModel(t, at.baseURL, name, opts...)
			if !errors.As(err, &streamOnly) || body != nil {
				t.Errorf("%s: the model only thinks, and only on a stream, got %v", name, err)
			}
		}
		if body, _ := callWithoutModel(t, at.baseURL, name, stream); body["stream"] != true {
			t.Errorf("%s: a streaming call must go out, got %v", name, body)
		}

		for _, model := range []string{"qwen3.9-27b", "qwen3.7-27b", "qwen3.5-1t"} {
			name = at.prefix + model
			body, err := callWithoutModel(t, at.baseURL, name)
			if err != nil || body == nil || [2]any{body["enable_thinking"], body["reasoning_effort"]} != [2]any{} {
				t.Errorf("%s: Model Studio documents no such open weights, so the call must go out as asked, got %v, %v",
					name, body, err)
			}
		}
	}
}
