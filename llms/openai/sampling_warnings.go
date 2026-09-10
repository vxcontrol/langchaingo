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
	warn.AddIntChange("WithTopK", model, reason, s.topK, req.TopK)
	warn.AddFloatChange("WithMinP", model, "the model rejects min_p", s.minP, req.MinP)
	warn.AddFloatChange("WithFrequencyPenalty", model, reason, s.frequencyPenalty, req.FrequencyPenalty)
	warn.AddFloatChange("WithPresencePenalty", model, reason, s.presencePenalty, req.PresencePenalty)

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
