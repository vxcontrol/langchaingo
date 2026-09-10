package bedrock

import (
	"strconv"

	"github.com/vxcontrol/langchaingo/llms"
)

// unreadBedrockOptions covers both paths: these options reach neither the legacy
// payloads nor ConverseInput.
func unreadBedrockOptions(model string, opts llms.CallOptions) []llms.Warning {
	const unread = "neither bedrock request has a field for it"

	var warnings []llms.Warning
	drop := func(option, asked string) {
		warnings = append(warnings, llms.Warning{
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
	return warnings
}
