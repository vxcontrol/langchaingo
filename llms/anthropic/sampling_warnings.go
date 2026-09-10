package anthropic

import (
	"strconv"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic/internal/anthropicclient"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func reportAnthropicUnread(warn *llms.Warnings, model string, opts llms.CallOptions) {
	const unread = "the door builds no field for it"

	drop := func(option, asked string) {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: option, Model: model,
			Asked: asked, Reason: unread,
		})
	}
	for _, o := range []struct {
		option string
		value  *float64
	}{
		{"WithMinP", opts.MinP},
		{"WithRepetitionPenalty", opts.RepetitionPenalty},
		{"WithFrequencyPenalty", opts.FrequencyPenalty},
		{"WithPresencePenalty", opts.PresencePenalty},
	} {
		if o.value != nil && *o.value != 0 {
			drop(o.option, strconv.FormatFloat(*o.value, 'g', -1, 64))
		}
	}
	for _, o := range []struct {
		option  string
		value   *int
		neutral int
	}{
		{"WithN", opts.N, 1},
		{"WithCandidateCount", opts.CandidateCount, 1},
		{"WithTopLogProbs", opts.TopLogProbs, 0},
	} {
		if o.value != nil && *o.value != o.neutral {
			drop(o.option, strconv.Itoa(*o.value))
		}
	}
	if opts.LogProbs != nil && *opts.LogProbs {
		drop("WithLogProbs", "true")
	}
}

func reportAnthropicMechanism(warn *llms.Warnings, model string, opts llms.CallOptions, thinking *anthropicclient.ThinkingPayload) {
	cfg := opts.Reasoning
	if cfg == nil || !cfg.Adaptive || thinking == nil || thinking.Type == "adaptive" {
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningSubstitute, Option: "WithAdaptiveReasoning", Model: model,
		Asked: "adaptive", Sent: thinking.Type,
		Reason: "the door takes the thinking mechanism from the model, not from the preference",
	})
}

func reportAnthropicBudget(warn *llms.Warnings, model string, opts llms.CallOptions, thinking *anthropicclient.ThinkingPayload) {
	cfg := opts.Reasoning
	if cfg == nil || !cfg.HasExplicitTokens() {
		return
	}
	sent := 0
	if thinking != nil {
		sent = thinking.Budget
	}
	if sent == cfg.Tokens {
		return
	}
	if sent <= 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: model,
			Asked:  strconv.Itoa(cfg.Tokens) + " tokens",
			Reason: "the door sends no thinking budget on this model",
		})
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningClamp, Option: "WithReasoning", Model: model,
		Asked: strconv.Itoa(cfg.Tokens) + " tokens", Sent: strconv.Itoa(sent) + " tokens",
		Reason: "the thinking budget is capped by the answer limit and by what the model records",
	})
}

func reportAnthropicSampling(
	warn *llms.Warnings, model string, opts llms.CallOptions,
	thinking *anthropicclient.ThinkingPayload,
	temperature, topP *float64, topK *int, maxTokens int,
) {
	reportAnthropicUnread(warn, model, opts)
	reportAnthropicBudget(warn, model, opts, thinking)
	reportAnthropicMechanism(warn, model, opts, thinking)

	reason := anthropicSamplingReason(model, thinking)
	warn.AddFloatChange("WithTemperature", model, reason, opts.Temperature, temperature)
	warn.AddFloatChange("WithTopP", model, reason, opts.TopP, topP)
	warn.AddIntChange("WithTopK", model, reason, opts.TopK, topK)

	if asked := opts.GetMaxTokens(); asked > 0 && maxTokens != asked {
		warn.Add(llms.Warning{
			Kind: llms.WarningClamp, Option: "WithMaxTokens", Model: model,
			Asked: strconv.Itoa(asked), Sent: strconv.Itoa(maxTokens),
			Reason: "the answer limit was raised to leave room for the thinking budget",
		})
	}
}

func anthropicSamplingReason(model string, thinking *anthropicclient.ThinkingPayload) string {
	switch {
	case reasoning.ClaudeRejectsSampling(model):
		return "the model rejects sampling parameters"
	case thinking != nil && thinking.Type == "adaptive":
		return "adaptive thinking takes no sampling parameters"
	case thinking != nil && thinking.Type == "enabled":
		return "the model refuses sampling while thinking"
	default:
		return "the model does not accept this combination of sampling parameters"
	}
}
