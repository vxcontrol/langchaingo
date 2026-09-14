package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func captureDeepSeekRequest(t *testing.T, model string, opts ...llms.CallOption) map[string]any {
	t.Helper()

	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("decode request: %v", err)
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
	if _, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return body
}

func TestDeepSeekV32KeepsCallerSamplingUntilThinkingIsAsked(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"deepseek-v3.2", "deepseek.v3.2", "us.deepseek.v3.2", "deepseek-v3.2-exp"} {
		body := captureDeepSeekRequest(t, model, llms.WithTemperature(0.25), llms.WithTopP(0.3))

		if got, want := body["temperature"], 0.25; got != want {
			t.Errorf("%s: temperature = %v, want %v", model, got, want)
		}
		if got, ok := body["top_p"]; !ok || got != 0.3 {
			t.Errorf("%s: top_p = %v (present %v), want 0.3", model, got, ok)
		}
	}
}

func TestDeepSeekV32KeepsSamplingEvenWithAnEffortOnTheWire(t *testing.T) {
	t.Parallel()

	body := captureDeepSeekRequest(t, "deepseek-v3.2",
		llms.WithTemperature(0.25), llms.WithTopP(0.3),
		llms.WithReasoning(llms.ReasoningHigh, 0))

	if got, want := body["temperature"], 0.25; got != want {
		t.Errorf("temperature = %v, want %v", got, want)
	}
	if got, ok := body["top_p"]; !ok || got != 0.3 {
		t.Errorf("top_p = %v (present %v), want 0.3", got, ok)
	}
	if got, want := body["reasoning_effort"], "high"; got != want {
		t.Errorf("reasoning_effort = %v, want %v", got, want)
	}
}

func TestDeepSeekThinkingLeavesOutTheTemperatureItIgnores(t *testing.T) {
	t.Parallel()

	sampling := []llms.CallOption{llms.WithTemperature(0.4), llms.WithTopP(0.97)}
	requests := map[string][]llms.CallOption{
		"deepseek-v4-pro":                    sampling,
		"deepseek/deepseek-v4-pro":           sampling,
		"deepseek-v4-flash":                  sampling,
		"deepseek-v4-pro with an effort":     append([]llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}, sampling...),
		"deepseek-v4-pro thinking by object": append([]llms.CallOption{thinkingInExtraBody("enabled")}, sampling...),
	}
	for name, opts := range requests {
		model, _, _ := strings.Cut(name, " ")

		body := captureDeepSeekRequest(t, model, opts...)
		if _, ok := body["temperature"]; ok {
			t.Errorf("%s: thinking mode ignores temperature, got body: %v", name, body)
		}
		if body["top_p"] != 0.97 {
			t.Errorf("%s: top_p takes effect while thinking and must stay, got body: %v", name, body)
		}

		resp := sendForWarnings(t, model, opts...)
		w := warningFor(t, resp, "WithTemperature")
		if w.Kind != llms.WarningDrop || w.Asked != "0.4" || !strings.Contains(w.Reason, "ignores temperature") {
			t.Errorf("%s: temperature warning = %+v", name, w)
		}
		for _, other := range resp.Warnings {
			if other.Option == "WithTopP" {
				t.Errorf("%s: top_p reached the wire, yet it is reported: %+v", name, other)
			}
		}
	}
}

func TestDeepSeekKeepsTheTemperatureOnceThinkingIsOff(t *testing.T) {
	t.Parallel()

	for name, off := range map[string]llms.CallOption{
		"reasoning disabled":          llms.WithReasoningDisabled(),
		"thinking object in the body": thinkingInExtraBody("disabled"),
		"effort none in the body":     llms.WithExtraBody(map[string]any{"reasoning_effort": "none"}),
	} {
		opts := []llms.CallOption{off, llms.WithTemperature(0.4)}
		if body := captureDeepSeekRequest(t, "deepseek-v4-pro", opts...); body["temperature"] != 0.4 {
			t.Errorf("%s: non-thinking mode takes temperature, got body: %v", name, body)
		}
		if resp := sendForWarnings(t, "deepseek-v4-pro", opts...); len(resp.Warnings) != 0 {
			t.Errorf("%s: nothing was lost, got %v", name, resp.Warnings)
		}
	}
}

func TestDeepSeekOnAnotherHostKeepsItsTemperatureWhileThinking(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"dashscope/deepseek-v4-pro", "openrouter/deepseek/deepseek-v4-pro"} {
		body := captureDeepSeekRequest(t, model, llms.WithTemperature(0.4), llms.WithReasoning(llms.ReasoningHigh, 0))
		if body["temperature"] != 0.4 {
			t.Errorf("%s: only DeepSeek's own API documents temperature as ignored while thinking, got body: %v",
				model, body)
		}
	}
}

func TestDeepSeekNeverGetsThePenaltiesItNoLongerSupports(t *testing.T) {
	t.Parallel()

	penalties := []llms.CallOption{llms.WithFrequencyPenalty(0.5), llms.WithPresencePenalty(0.3)}
	requests := map[string][]llms.CallOption{
		"deepseek-v4-pro":              penalties,
		"deepseek/deepseek-v4-pro":     penalties,
		"deepseek-flash":               penalties,
		"dashscope/deepseek-v4-pro":    penalties,
		"deepseek-v4-pro thinking off": append([]llms.CallOption{llms.WithReasoningDisabled()}, penalties...),
	}
	for name, opts := range requests {
		model, _, _ := strings.Cut(name, " ")

		body := captureDeepSeekRequest(t, model, opts...)
		for _, field := range []string{"frequency_penalty", "presence_penalty"} {
			if _, ok := body[field]; ok {
				t.Errorf("%s: %s takes no effect on DeepSeek, got body: %v", name, field, body)
			}
		}

		resp := sendForWarnings(t, model, opts...)
		for option, asked := range map[string]string{"WithFrequencyPenalty": "0.5", "WithPresencePenalty": "0.3"} {
			if w := warningFor(t, resp, option); w.Kind != llms.WarningDrop || w.Asked != asked {
				t.Errorf("%s: %s warning = %+v", name, option, w)
			}
		}
	}
}

func thinkingInExtraBody(kind string) llms.CallOption {
	return llms.WithExtraBody(map[string]any{"thinking": map[string]any{"type": kind}})
}

func TestChatVariantsKeepTheTemperatureTheCallerSet(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-5.2-chat", "gpt-5.2-chat-latest", "gpt-5-chat"} {
		body := captureDeepSeekRequest(t, model,
			llms.WithTemperature(0.3), llms.WithReasoning(llms.ReasoningHigh, 0))
		if body["temperature"] != 0.3 {
			t.Errorf("%s must keep the caller's temperature, got body: %v", model, body)
		}
	}
}

func TestPenaltiesStayOffTheDoorsThatRefuseThem(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"grok-4.6", "grok-4-1-fast", "grok-4-1-fast-non-reasoning", "grok-3", "grok-3-mini",
		"grok-3-mini-beta", "grok-4.20-0309-non-reasoning", "xai/grok-4.6",
	} {
		body := captureDeepSeekRequest(t, model,
			llms.WithFrequencyPenalty(0.5), llms.WithPresencePenalty(0.5))
		if _, ok := body["frequency_penalty"]; ok {
			t.Errorf("%s answers that it does not support the parameter, got body: %v", model, body)
		}
		if _, ok := body["presence_penalty"]; ok {
			t.Errorf("%s answers that it does not support the parameter, got body: %v", model, body)
		}
	}

	for _, model := range []string{"gpt-5.4", "glm-5.2", "mistral-medium-latest"} {
		body := captureDeepSeekRequest(t, model,
			llms.WithFrequencyPenalty(0.5), llms.WithPresencePenalty(0.5))
		if body["frequency_penalty"] != 0.5 || body["presence_penalty"] != 0.5 {
			t.Errorf("%s takes both penalties and must keep receiving them, got body: %v", model, body)
		}
	}
}

func TestStopStillTravelsToTheDoorsThatRefuseIt(t *testing.T) {
	t.Parallel()

	body := captureDeepSeekRequest(t, "grok-4.6", llms.WithStopWords([]string{"STOP"}))
	if _, ok := body["stop"]; !ok {
		t.Fatalf("stop bounds the answer, so it is left to fail loudly rather than dropped, got body: %v", body)
	}
}

func TestLogProbsReachTheWireAndYieldToThinking(t *testing.T) {
	t.Parallel()

	t.Run("asked", func(t *testing.T) {
		t.Parallel()

		body := captureDeepSeekRequest(t, "gpt-4.1", llms.WithLogProbs(true), llms.WithTopLogProbs(3))
		if body["logprobs"] != true {
			t.Errorf("logprobs must reach the wire, got body: %v", body)
		}
		if body["top_logprobs"] != float64(3) {
			t.Errorf("top_logprobs must reach the wire, got body: %v", body)
		}
	})

	t.Run("not asked", func(t *testing.T) {
		t.Parallel()

		body := captureDeepSeekRequest(t, "gpt-4.1")
		if _, ok := body["logprobs"]; ok {
			t.Errorf("an unset logprobs must stay off the wire, got body: %v", body)
		}
		if _, ok := body["top_logprobs"]; ok {
			t.Errorf("an unset top_logprobs must stay off the wire, got body: %v", body)
		}
	})

	t.Run("a thinking generation drops them like the sampling params", func(t *testing.T) {
		t.Parallel()

		body := captureDeepSeekRequest(t, "gpt-5.4",
			llms.WithLogProbs(true), llms.WithTopLogProbs(3), llms.WithReasoning(llms.ReasoningHigh, 0))
		if _, ok := body["logprobs"]; ok {
			t.Errorf("gpt-5.4 takes logprobs only at the none effort, got body: %v", body)
		}
		if _, ok := body["top_logprobs"]; ok {
			t.Errorf("gpt-5.4 takes top_logprobs only at the none effort, got body: %v", body)
		}
	})
}
