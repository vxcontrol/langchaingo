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

func sendForWarnings(t *testing.T, model string, opts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	const completion = `{"id":"x","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completion)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithBaseURL(srv.URL), WithToken("test"), WithModel(model))
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	if err != nil {
		t.Fatalf("GenerateContent() error: %v", err)
	}
	return resp
}

func warningFor(t *testing.T, resp *llms.ContentResponse, option string) llms.Warning {
	t.Helper()

	for _, w := range resp.Warnings {
		if w.Option == option {
			return w
		}
	}
	t.Fatalf("no warning for %s, got %v", option, resp.Warnings)
	return llms.Warning{}
}

func TestTheCallerLearnsWhatTheSamplingPolicyTookAway(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "gpt-5",
		llms.WithTemperature(0.2), llms.WithTopP(0.9),
		llms.WithLogProbs(true), llms.WithTopLogProbs(3))

	temperature := warningFor(t, resp, "WithTemperature")
	if temperature.Kind != llms.WarningSubstitute || temperature.Asked != "0.2" || temperature.Sent != "1" {
		t.Errorf("temperature warning = %+v", temperature)
	}
	if temperature.Model != "gpt-5" {
		t.Errorf("temperature warning names model %q", temperature.Model)
	}

	for _, option := range []string{"WithTopP", "WithLogProbs", "WithTopLogProbs"} {
		if got := warningFor(t, resp, option); got.Kind != llms.WarningDrop || got.Sent != "" {
			t.Errorf("%s warning = %+v", option, got)
		}
	}
}

func TestAnUntouchedRequestCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "gpt-4o", llms.WithTemperature(0.2), llms.WithTopP(0.9))
	if len(resp.Warnings) != 0 {
		t.Errorf("want no warnings on a model that keeps sampling, got %v", resp.Warnings)
	}
}

func TestPenaltiesRefusedByTheFamilyAreReported(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "grok-4",
		llms.WithFrequencyPenalty(0.7), llms.WithPresencePenalty(0.4))

	for option, asked := range map[string]string{
		"WithFrequencyPenalty": "0.7", "WithPresencePenalty": "0.4",
	} {
		w := warningFor(t, resp, option)
		if w.Kind != llms.WarningDrop || w.Asked != asked || w.Sent != "" {
			t.Errorf("%s warning = %+v", option, w)
		}
	}
}

func TestAnEffortLoweredToWhatTheModelTakesIsReported(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "gpt-5.1", llms.WithReasoning(llms.ReasoningXHigh, 0))

	w := warningFor(t, resp, "WithReasoning")
	if w.Kind != llms.WarningClamp || w.Asked != "xhigh" || w.Sent != "high" {
		t.Errorf("reasoning warning = %+v", w)
	}
}

func TestAnEffortOnAModelThatSendsNoneIsReported(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "gpt-4o", llms.WithReasoning(llms.ReasoningHigh, 0))

	w := warningFor(t, resp, "WithReasoning")
	if w.Kind != llms.WarningDrop || w.Asked != "high" || w.Sent != "" {
		t.Errorf("reasoning warning = %+v", w)
	}
}

func TestAnAnswerLimitRaisedForTheBudgetIsReported(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "claude-sonnet-4-5",
		llms.WithMaxTokens(1000), llms.WithReasoning(llms.ReasoningMedium, 4096))

	w := warningFor(t, resp, "WithMaxTokens")
	if w.Kind != llms.WarningClamp || w.Asked != "1000" {
		t.Errorf("max-tokens warning = %+v", w)
	}
	if w.Sent == w.Asked {
		t.Errorf("max-tokens warning reports no change: %+v", w)
	}
}

func TestAThinkingBudgetCutToFitTheAnswerLimitIsReported(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "qwen3-max",
		llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningNone, 30000))

	w := warningFor(t, resp, "WithReasoning")
	if w.Kind != llms.WarningClamp || w.Asked != "30000 tokens" {
		t.Fatalf("reasoning warning = %+v (all: %v)", w, resp.Warnings)
	}
	if w.Sent == w.Asked || w.Sent == "" {
		t.Errorf("reasoning warning reports no cut: %+v", w)
	}
}

func TestABudgetThisDoorPutsOnNoFieldIsReportedAsDropped(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "claude-sonnet-4-5",
		llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningNone, 30000))

	w := warningFor(t, resp, "WithReasoning")
	if w.Kind != llms.WarningDrop || w.Asked != "30000 tokens" || w.Sent != "" {
		t.Errorf("reasoning warning = %+v (all: %v)", w, resp.Warnings)
	}
}

func TestAnEffortReplacedByABudgetIsReported(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "qwen3.8-plus",
		llms.WithMaxTokens(8192), llms.WithReasoning(llms.ReasoningHigh, 4096))

	w := warningFor(t, resp, "WithReasoning")
	if w.Kind != llms.WarningDrop || w.Asked != "high" || w.Sent != "" {
		t.Errorf("reasoning warning = %+v (all: %v)", w, resp.Warnings)
	}
}

func TestAZeroPenaltyIsNotALoss(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "grok-4",
		llms.WithFrequencyPenalty(0), llms.WithPresencePenalty(0))

	if len(resp.Warnings) != 0 {
		t.Errorf("no penalty was asked for, got %v", resp.Warnings)
	}
}

func TestAZeroTopKIsNotALoss(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "gpt-5", llms.WithTopK(0))
	if len(resp.Warnings) != 0 {
		t.Errorf("no top-k was asked for, got %v", resp.Warnings)
	}
}

func TestOptionsThisDoorReadsNowhereAreReported(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "gpt-4o",
		llms.WithCandidateCount(3), llms.WithMinLength(10),
		llms.WithMaxLength(4096), llms.WithResponseMIMEType("application/json"))

	for option, asked := range map[string]string{
		"WithCandidateCount": "3", "WithMinLength": "10",
		"WithMaxLength": "4096", "WithResponseMIMEType": "application/json",
	} {
		w := warningFor(t, resp, option)
		if w.Kind != llms.WarningDrop || w.Asked != asked {
			t.Errorf("%s warning = %+v", option, w)
		}
	}
}

func TestFieldsThisEndpointRefusesStayOffTheWire(t *testing.T) {
	t.Parallel()

	body := sendForWire(t, "gpt-4o", llms.WithTopK(7), llms.WithRepetitionPenalty(1.1))

	for _, field := range []string{`"top_k"`, `"repetition_penalty"`} {
		if strings.Contains(body, field) {
			t.Errorf("%s reached the wire; OpenAI answers 400 on it: %s", field, body)
		}
	}
}

func TestAFieldTheEndpointRefusesIsReported(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "gpt-4o", llms.WithTopK(7), llms.WithRepetitionPenalty(1.1))

	for option, asked := range map[string]string{"WithTopK": "7", "WithRepetitionPenalty": "1.1"} {
		w := warningFor(t, resp, option)
		if w.Kind != llms.WarningDrop || w.Asked != asked {
			t.Errorf("%s warning = %+v", option, w)
		}
	}
}

func TestAVendorThatTakesTopKStillGetsIt(t *testing.T) {
	t.Parallel()

	body := sendForWire(t, "zai/glm-4.5-air", llms.WithTopK(7))

	if !strings.Contains(body, `"top_k":7`) {
		t.Errorf("glm takes top_k and answered 200 to it; the wire lost it: %s", body)
	}
}

func TestTheDoorThatMergesExtraBodyReportsNoLoss(t *testing.T) {
	t.Parallel()

	extra := llms.WithExtraBody(map[string]any{"enable_thinking": false})

	body := sendForWire(t, "gpt-4o", extra)
	if !strings.Contains(body, `"enable_thinking":false`) {
		t.Errorf("extra body did not reach the wire: %s", body)
	}

	resp := sendForWarnings(t, "gpt-4o", extra)
	if len(resp.Warnings) != 0 {
		t.Errorf("this door merges extra body, nothing is lost, got %v", resp.Warnings)
	}
}

func TestTheNeutralOptionAndTheDoorOptionAreTheSame(t *testing.T) {
	t.Parallel()

	fields := map[string]any{"enable_thinking": false}
	for name, opt := range map[string]llms.CallOption{
		"neutral": llms.WithExtraBody(fields),
		"door":    WithExtraBody(fields),
	} {
		if body := sendForWire(t, "gpt-4o", opt); !strings.Contains(body, `"enable_thinking":false`) {
			t.Errorf("%s option lost the fields: %s", name, body)
		}
	}
}

func TestAVendorThatKeepsSamplingWhileThinkingKeepsIt(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"glm-5.3", "zai/glm-5.3", "kimi-k3"} {
		body := sendForWire(t, model,
			llms.WithTemperature(0.2), llms.WithTopP(0.9),
			llms.WithReasoning(llms.ReasoningHigh, 0))

		if !strings.Contains(body, `"temperature":0.2`) {
			t.Errorf("%s: the caller's temperature did not reach the wire: %s", model, body)
		}
		if !strings.Contains(body, `"top_p":0.9`) {
			t.Errorf("%s: the caller's top_p did not reach the wire: %s", model, body)
		}
	}
}

func TestOpenAIsOwnReasoningModelsStillLoseTheirSampling(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-5.1", "o3-mini", "gpt-5-pro"} {
		body := sendForWire(t, model,
			llms.WithTemperature(0.2), llms.WithTopP(0.9),
			llms.WithReasoning(llms.ReasoningHigh, 0))

		if strings.Contains(body, `"temperature":0.2`) {
			t.Errorf("%s: this endpoint refuses a sampling temperature while thinking: %s", model, body)
		}
		if strings.Contains(body, `"top_p"`) {
			t.Errorf("%s: this endpoint refuses top_p while thinking: %s", model, body)
		}
	}
}

func TestClaudeOnThisDoorStillLosesItsSampling(t *testing.T) {
	t.Parallel()

	body := sendForWire(t, "claude-sonnet-4-5",
		llms.WithTemperature(0.2), llms.WithTopP(0.9),
		llms.WithReasoning(llms.ReasoningHigh, 4096))

	if strings.Contains(body, `"temperature":0.2`) {
		t.Errorf("thinking claude takes temperature 1: %s", body)
	}
}
