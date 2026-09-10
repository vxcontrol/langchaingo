package anthropic

import (
	"strconv"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic/internal/anthropicclient"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func reportAnthropicSampling(
	warn *llms.Warnings, model string, opts llms.CallOptions,
	thinking *anthropicclient.ThinkingPayload,
	temperature, topP *float64, topK *int, maxTokens int,
) {
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
