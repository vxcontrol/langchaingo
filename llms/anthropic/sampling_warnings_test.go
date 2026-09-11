package anthropic_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
)

const warningsTestModel = "claude-sonnet-4-5"

func generateForWarnings(t *testing.T, callOpts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()
	return generateForModel(t, warningsTestModel, callOpts...)
}

func generateForModel(t *testing.T, model string, callOpts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_test","type":"message","role":"assistant",` +
			`"model":"` + model + `","content":[{"type":"text","text":"ok"}],` +
			`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(
		anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL),
		anthropic.WithModel(model),
	)
	require.NoError(t, err)

	messages := []llms.MessageContent{{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.TextPart("hi")},
	}}
	resp, err := llm.GenerateContent(t.Context(), messages, callOpts...)
	require.NoError(t, err)
	return resp
}

func warningsByOption(warnings []llms.Warning) map[string]llms.Warning {
	byOption := make(map[string]llms.Warning, len(warnings))
	for _, w := range warnings {
		byOption[w.Option] = w
	}
	return byOption
}

func warningsFor(warnings []llms.Warning, option string) []llms.Warning {
	var found []llms.Warning
	for _, w := range warnings {
		if w.Option == option {
			found = append(found, w)
		}
	}
	return found
}

func TestThinkingTakesTheSamplingParamsAndSaysSo(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithReasoning(llms.ReasoningMedium, 4096),
		llms.WithMaxTokens(4096),
		llms.WithTemperature(0.2),
		llms.WithTopP(0.9),
		llms.WithTopK(40),
	)

	got := warningsByOption(resp.Warnings)

	temperature, ok := got["WithTemperature"]
	require.True(t, ok, "no temperature warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningSubstitute, temperature.Kind)
	require.Equal(t, "0.2", temperature.Asked)
	require.Equal(t, "1", temperature.Sent)

	for _, option := range []string{"WithTopP", "WithTopK"} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, llms.WarningDrop, w.Kind)
		require.Empty(t, w.Sent)
	}

	require.NotContains(t, got, "WithMaxTokens",
		"the answer limit fits the budget here, so nothing should be raised")
}

func TestAnAnswerLimitRaisedForTheBudgetIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithReasoning(llms.ReasoningMedium, 4096),
		llms.WithMaxTokens(1000),
	)

	limit, ok := warningsByOption(resp.Warnings)["WithMaxTokens"]
	require.True(t, ok, "no max-tokens warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, limit.Kind)
	require.Equal(t, "1000", limit.Asked)
	require.Equal(t, "2048", limit.Sent)
}

func TestARequestThatKeepsItsSamplingCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithTemperature(0.2),
		llms.WithMaxTokens(4096),
	)
	require.Empty(t, resp.Warnings)
}

func TestOptionsThisDoorBuildsNoFieldForAreReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithMinP(0.05), llms.WithRepetitionPenalty(1.1),
		llms.WithFrequencyPenalty(0.7), llms.WithPresencePenalty(0.25),
		llms.WithN(3), llms.WithCandidateCount(2),
		llms.WithLogProbs(true), llms.WithTopLogProbs(4),
	)

	got := warningsByOption(resp.Warnings)
	for option, asked := range map[string]string{
		"WithMinP": "0.05", "WithRepetitionPenalty": "1.1",
		"WithFrequencyPenalty": "0.7", "WithPresencePenalty": "0.25",
		"WithN": "3", "WithCandidateCount": "2",
		"WithLogProbs": "true", "WithTopLogProbs": "4",
	} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, llms.WarningDrop, w.Kind)
		require.Equal(t, asked, w.Asked)
	}
}

func TestAThinkingBudgetCutToFitTheAnswerLimitIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningMedium, 30000))

	var clamp *llms.Warning
	for _, w := range warningsFor(resp.Warnings, "WithReasoning") {
		if w.Kind == llms.WarningClamp {
			clamp = &w
		}
	}
	require.NotNil(t, clamp, "no budget clamp in %v", resp.Warnings)
	require.Equal(t, "30000 tokens", clamp.Asked)
	require.NotEqual(t, clamp.Asked, clamp.Sent)
}

func TestAThinkingMechanismTheModelDoesNotOfferIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithAdaptiveReasoning(llms.ReasoningHigh))

	w, ok := warningsByOption(resp.Warnings)["WithAdaptiveReasoning"]
	require.True(t, ok, "no mechanism warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningSubstitute, w.Kind)
	require.Equal(t, "adaptive", w.Asked)
	require.Equal(t, "enabled", w.Sent)
}

func TestAskingForWhatTheDoorAlreadyDoesIsNotALoss(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithMaxTokens(4096),
		llms.WithLogProbs(false), llms.WithN(1), llms.WithCandidateCount(1),
		llms.WithTopLogProbs(0), llms.WithFrequencyPenalty(0),
	)

	require.Empty(t, resp.Warnings,
		"a single choice with no logprobs and no penalty is what the door sends anyway")
}

func TestAnEffortLoweredToWhatTheModelTakesIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForModel(t, "claude-sonnet-4-6",
		llms.WithMaxTokens(8192), llms.WithAdaptiveReasoning(llms.ReasoningXHigh))

	w, ok := warningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, w.Kind)
	require.Equal(t, "xhigh", w.Asked)
	require.Equal(t, "high", w.Sent)
}

func TestAskingForJSONOnADoorThatSendsNoneIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithJSONMode())

	w, ok := warningsByOption(resp.Warnings)["WithJSONMode"]
	require.True(t, ok, "no json-mode warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "true", w.Asked)
}

func TestExtraBodyThisDoorCannotMergeIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithExtraBody(map[string]any{"enable_thinking": false, "chat_template_kwargs": map[string]any{}}))

	w, ok := warningsByOption(resp.Warnings)["WithExtraBody"]
	require.True(t, ok, "no extra-body warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "chat_template_kwargs, enable_thinking", w.Asked)
}
