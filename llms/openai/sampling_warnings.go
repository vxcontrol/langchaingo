package openai

import (
	"strconv"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai/internal/openaiclient"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type warnCtx struct {
	model string
	sink  *llms.Warnings
}

type samplingSnapshot struct {
	temperature      *float64
	topP             *float64
	topK             *int
	minP             *float64
	frequencyPenalty *float64
	presencePenalty  *float64
	logProbs         bool
	topLogProbs      int
}

func takeSamplingSnapshot(req *openaiclient.ChatRequest) samplingSnapshot {
	return samplingSnapshot{
		temperature:      req.Temperature,
		topP:             req.TopP,
		topK:             req.TopK,
		minP:             req.MinP,
		frequencyPenalty: req.FrequencyPenalty,
		presencePenalty:  req.PresencePenalty,
		logProbs:         req.LogProbs,
		topLogProbs:      req.TopLogProbs,
	}
}

func (s samplingSnapshot) report(req *openaiclient.ChatRequest, model, reason string, warn *llms.Warnings) {
	warn.AddFloatChange("WithTemperature", model, reason, s.temperature, req.Temperature)
	warn.AddFloatChange("WithTopP", model, reason, s.topP, req.TopP)
	addNonZeroIntChange(warn, "WithTopK", model, reason, s.topK, req.TopK)
	addNonZeroChange(warn, "WithMinP", model, "the model rejects min_p", s.minP, req.MinP)
	addNonZeroChange(warn, "WithFrequencyPenalty", model, reason, s.frequencyPenalty, req.FrequencyPenalty)
	addNonZeroChange(warn, "WithPresencePenalty", model, reason, s.presencePenalty, req.PresencePenalty)

	if s.logProbs && !req.LogProbs {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithLogProbs", Model: model,
			Asked: "true", Reason: reason,
		})
	}
	if s.topLogProbs > 0 && req.TopLogProbs != s.topLogProbs {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTopLogProbs", Model: model,
			Asked: strconv.Itoa(s.topLogProbs), Reason: reason,
		})
	}
}

// addNonZeroChange is for options whose zero asks for nothing — unlike temperature.
const refusedByEndpoint = "the endpoint refuses the whole request when this field is present"

func addNonZeroChange(warn *llms.Warnings, option, model, reason string, before, after *float64) {
	if before == nil || *before == 0 {
		return
	}
	warn.AddFloatChange(option, model, reason, before, after)
}

func addNonZeroIntChange(warn *llms.Warnings, option, model, reason string, before, after *int) {
	if before == nil || *before == 0 {
		return
	}
	warn.AddIntChange(option, model, reason, before, after)
}

func reportOpenAIUnread(warn *llms.Warnings, model string, opts llms.CallOptions) {
	const unread = "the door builds no field for it"

	drop := func(option, asked string) {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: option, Model: model,
			Asked: asked, Reason: unread,
		})
	}
	for _, o := range []struct {
		option  string
		value   *int
		neutral int
	}{
		{"WithCandidateCount", opts.CandidateCount, 1},
		{"WithMinLength", opts.MinLength, 0},
		{"WithMaxLength", opts.MaxLength, 0},
	} {
		if o.value != nil && *o.value != o.neutral {
			drop(o.option, strconv.Itoa(*o.value))
		}
	}
	if opts.ResponseMIMEType != nil && *opts.ResponseMIMEType != "" {
		drop("WithResponseMIMEType", *opts.ResponseMIMEType)
	}
}

func samplingReason(model string, opts llms.CallOptions, wireEffort string) string {
	switch {
	case reasoning.ClaudeRejectsSampling(model):
		return "the model rejects sampling parameters"
	case refusesSamplingWhileThinking(model, opts, wireEffort):
		return "the model refuses sampling while thinking"
	default:
		return "the model does not accept this combination of sampling parameters"
	}
}

func reportOpenAIReasoning(warn *llms.Warnings, model string, cfg *llms.ReasoningConfig, req *openaiclient.ChatRequest) {
	if cfg == nil {
		return
	}
	effort, budget := "", 0
	if req.ReasoningEffort != nil {
		effort = string(*req.ReasoningEffort)
	}
	if req.ThinkingBudget != nil {
		budget = *req.ThinkingBudget
	}
	if req.Reasoning != nil {
		if req.Reasoning.Effort != "" {
			effort = string(req.Reasoning.Effort)
		}
		if req.Reasoning.MaxTokens > 0 {
			budget = req.Reasoning.MaxTokens
		}
	}
	reportOpenAIEffort(warn, model, cfg, effort)
	reportOpenAIBudget(warn, model, cfg, budget)
}

func reportOpenAIEffort(warn *llms.Warnings, model string, cfg *llms.ReasoningConfig, sent string) {
	asked := string(cfg.Effort)
	switch {
	case asked == "" || asked == string(llms.ReasoningNone):
		return
	case sent == "" || sent == reasoning.OpenAIDisableEffort:
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: model,
			Asked: asked, Reason: "the door sends no effort field on this model",
		})
	case asked != sent:
		warn.Add(llms.Warning{
			Kind: llms.WarningClamp, Option: "WithReasoning", Model: model,
			Asked: asked, Sent: sent,
			Reason: "the door sends only the efforts it records this model as accepting",
		})
	}
}

func reportOpenAIBudget(warn *llms.Warnings, model string, cfg *llms.ReasoningConfig, sent int) {
	if !cfg.HasExplicitTokens() || sent == cfg.Tokens {
		return
	}
	switch {
	case sent <= 0:
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: model,
			Asked:  strconv.Itoa(cfg.Tokens) + " tokens",
			Reason: "the door sends no thinking budget on this model",
		})
	default:
		warn.Add(llms.Warning{
			Kind: llms.WarningClamp, Option: "WithReasoning", Model: model,
			Asked: strconv.Itoa(cfg.Tokens) + " tokens", Sent: strconv.Itoa(sent) + " tokens",
			Reason: "the thinking budget is capped by the answer limit and by what the model records",
		})
	}
}
