package bedrock

import (
	"strconv"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock/internal/bedrockclient"
)

// The legacy payloads differ per provider; the Converse request has none of
// these fields at all.
var (
	legacyCarriesPenalties      = map[string]bool{"ai21": true}
	legacyCarriesCandidateCount = map[string]bool{"ai21": true, "cohere": true}
)

func unreadBedrockOptions(model string, converse bool, opts llms.CallOptions) []llms.Warning {
	const unread = "the bedrock request for this model has no field for it"

	provider := bedrockclient.GetProvider(model)
	carriesPenalties := !converse && legacyCarriesPenalties[provider]
	carriesCandidateCount := !converse && legacyCarriesCandidateCount[provider]

	var warnings []llms.Warning
	drop := func(option, asked string) {
		warnings = append(warnings, llms.Warning{
			Kind: llms.WarningDrop, Option: option, Model: model,
			Asked: asked, Reason: unread,
		})
	}
	for _, o := range []struct {
		option  string
		value   *float64
		carried bool
	}{
		{"WithMinP", opts.MinP, false},
		{"WithRepetitionPenalty", opts.RepetitionPenalty, carriesPenalties},
		{"WithFrequencyPenalty", opts.FrequencyPenalty, carriesPenalties},
		{"WithPresencePenalty", opts.PresencePenalty, carriesPenalties},
	} {
		if !o.carried && o.value != nil && *o.value != 0 {
			drop(o.option, strconv.FormatFloat(*o.value, 'g', -1, 64))
		}
	}
	for _, o := range []struct {
		option  string
		value   *int
		neutral int
		carried bool
	}{
		{"WithN", opts.N, 1, false},
		{"WithCandidateCount", opts.CandidateCount, 1, carriesCandidateCount},
		{"WithTopLogProbs", opts.TopLogProbs, 0, false},
	} {
		if !o.carried && o.value != nil && *o.value != o.neutral {
			drop(o.option, strconv.Itoa(*o.value))
		}
	}
	if opts.LogProbs != nil && *opts.LogProbs {
		drop("WithLogProbs", "true")
	}
	if opts.JSONMode {
		drop("WithJSONMode", "true")
	}
	return warnings
}
