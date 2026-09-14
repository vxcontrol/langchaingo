package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
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
	if temperature.Kind != llms.WarningDrop || temperature.Asked != "0.2" || temperature.Sent != "" {
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

	resp := sendForWarnings(t, "anthropic/claude-sonnet-5",
		llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningNone, 30000))

	w := warningFor(t, resp, "WithReasoning")
	if w.Kind != llms.WarningDrop || w.Asked != "30000 tokens" || w.Sent != "" {
		t.Errorf("reasoning warning = %+v (all: %v)", w, resp.Warnings)
	}
}

func TestABudgetTheThinkingObjectCarriesIsNotReported(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "claude-sonnet-4-5",
		llms.WithMaxTokens(8192), llms.WithReasoning(llms.ReasoningNone, 2048))

	if len(resp.Warnings) != 0 {
		t.Errorf("the budget reached the wire as asked, got %v", resp.Warnings)
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

	for _, model := range []string{"glm-5.3", "zai/glm-5.3", "kimi-k2-thinking"} {
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
	if strings.Contains(body, `"top_p"`) {
		t.Errorf("thinking claude takes no top_p below 0.95 or beside a temperature: %s", body)
	}
}

func TestMinPStaysOffOpenAIsOwnEndpoint(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-4o", "o3-mini", "gpt-5.1", "chatgpt-4o-latest"} {
		if body := sendForWire(t, model, llms.WithMinP(0.05)); strings.Contains(body, `"min_p"`) {
			t.Errorf("%s: this endpoint refuses the whole request on min_p: %s", model, body)
		}
	}

	resp := sendForWarnings(t, "gpt-4o", llms.WithMinP(0.05))
	w := warningFor(t, resp, "WithMinP")
	if w.Kind != llms.WarningDrop || w.Asked != "0.05" {
		t.Errorf("min-p warning = %+v", w)
	}
}

func TestAVendorThatTakesMinPStillGetsIt(t *testing.T) {
	t.Parallel()

	body := sendForWire(t, "zai/glm-4.5-air", llms.WithMinP(0.05))
	if !strings.Contains(body, `"min_p":0.05`) {
		t.Errorf("this vendor takes min_p; the wire lost it: %s", body)
	}
}

func TestAZeroMinPIsNotALoss(t *testing.T) {
	t.Parallel()

	if resp := sendForWarnings(t, "gpt-4o", llms.WithMinP(0)); len(resp.Warnings) != 0 {
		t.Errorf("no min-p was asked for, got %v", resp.Warnings)
	}
}

func TestMetadataSetAfterExtraBodyDoesNotDiscardIt(t *testing.T) {
	t.Parallel()

	fields := map[string]any{"enable_thinking": false}
	orders := map[string][]llms.CallOption{
		"extra body first": {llms.WithExtraBody(fields), llms.WithMetadata(map[string]any{"user": "u1"})},
		"metadata first":   {llms.WithMetadata(map[string]any{"user": "u1"}), llms.WithExtraBody(fields)},
	}
	for name, opts := range orders {
		if body := sendForWire(t, "gpt-4o", opts...); !strings.Contains(body, `"enable_thinking":false`) {
			t.Errorf("%s: the extra body did not reach the wire: %s", name, body)
		}
	}
}

func TestAdaptiveWithToolsNoLongerRefusesTheRequest(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5"} {
		body := sendForWire(t, model,
			llms.WithAdaptiveReasoning(llms.ReasoningNone), llms.WithTools([]llms.Tool{weatherTool()}))

		if strings.Contains(body, `"reasoning_effort"`) {
			t.Errorf("%s: the vendor serves an effort with tools on another API only: %s", model, body)
		}
		if !strings.Contains(body, `"tools"`) {
			t.Errorf("%s: the request lost its tools: %s", model, body)
		}
	}
}

func TestAnEffortNamedAlongsideAdaptiveStillTravels(t *testing.T) {
	t.Parallel()

	body := sendForWire(t, "gpt-5.1", llms.WithAdaptiveReasoning(llms.ReasoningLow))
	if !strings.Contains(body, `"reasoning_effort":"low"`) {
		t.Errorf("a named effort is a depth, not a hand-off: %s", body)
	}
}

func TestDelegatedDepthOnAModelThatWillNotReasonIsReported(t *testing.T) {
	t.Parallel()

	optIn := sendForWarnings(t, "gpt-5.1", llms.WithAdaptiveReasoning(llms.ReasoningNone))
	w := warningFor(t, optIn, "WithAdaptiveReasoning")
	if w.Kind != llms.WarningDrop || w.Asked != "adaptive" || w.Sent != "" {
		t.Errorf("opt-in model warning = %+v", w)
	}

	withTools := sendForWarnings(t, "gpt-5.6",
		llms.WithAdaptiveReasoning(llms.ReasoningNone), llms.WithTools([]llms.Tool{weatherTool()}))
	w = warningFor(t, withTools, "WithAdaptiveReasoning")
	if w.Kind != llms.WarningSubstitute || w.Sent != "none" {
		t.Errorf("forced-disable warning = %+v", w)
	}
}

func TestDelegatedDepthOnAModelThatReasonsAnywayIsSilent(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "gpt-4o", llms.WithAdaptiveReasoning(llms.ReasoningNone))
	if len(resp.Warnings) != 0 {
		t.Errorf("nothing was lost, got %v", resp.Warnings)
	}
}

func TestKimiModelsWithFixedSamplingGetNoneOfIt(t *testing.T) {
	t.Parallel()

	sampling := []llms.CallOption{
		llms.WithTemperature(0.3), llms.WithTopP(0.5),
		llms.WithPresencePenalty(0.2), llms.WithFrequencyPenalty(0.1),
	}
	requests := map[string][]llms.CallOption{
		"kimi-k3":                  sampling,
		"moonshot/kimi-k3":         sampling,
		"kimi-k2.7-code":           sampling,
		"kimi-k2.7-code-highspeed": sampling,
		"kimi-k2.6":                slices.Concat(sampling, []llms.CallOption{llms.WithReasoningDisabled()}),
	}
	for model, opts := range requests {
		body := sendForWire(t, model, opts...)
		for _, field := range []string{`"temperature"`, `"top_p"`, `"presence_penalty"`, `"frequency_penalty"`} {
			if strings.Contains(body, field) {
				t.Errorf("%s: %s is fixed by the vendor and must stay off the wire: %s", model, field, body)
			}
		}

		resp := sendForWarnings(t, model, sampling...)
		for _, option := range []string{"WithTemperature", "WithTopP", "WithPresencePenalty", "WithFrequencyPenalty"} {
			if got := warningFor(t, resp, option); got.Kind != llms.WarningDrop {
				t.Errorf("%s: %s warning = %+v", model, option, got)
			}
		}
	}
}

func TestKimiModelsOutsideTheFixedListKeepTheirSampling(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"kimi-k2-thinking", "kimi-k2.7", "kimi-k30"} {
		body := sendForWire(t, model, llms.WithTemperature(0.3), llms.WithTopP(0.5))
		if !strings.Contains(body, `"temperature":0.3`) || !strings.Contains(body, `"top_p":0.5`) {
			t.Errorf("%s: sampling the vendor does not fix must reach the wire: %s", model, body)
		}
	}
}
