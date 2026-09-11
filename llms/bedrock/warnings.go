package bedrock

import (
	"strconv"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock/internal/bedrockclient"
)

func legacyCarriesPenalties(model string) bool {
	return bedrockclient.GetProvider(model) == "ai21" && !bedrockclient.IsAi21Jamba(model)
}

func legacyCarriesCandidateCount(model string) bool {
	switch bedrockclient.GetProvider(model) {
	case "ai21":
		return true
	case "cohere":
		return !bedrockclient.IsCohereCommandR(model)
	}
	return false
}

func unreadBedrockOptions(model string, converse bool, opts llms.CallOptions) []llms.Warning {
	const unread = "the bedrock request for this model has no field for it"

	carriesPenalties := !converse && legacyCarriesPenalties(model)
	carriesCandidateCount := !converse && legacyCarriesCandidateCount(model)

	var warnings []llms.Warning
	var extra llms.Warnings
	extra.AddUnreadExtraBody(model, opts, extraBodyUnread)
	warnings = append(warnings, extra.List()...)

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
	if opts.JSONMode && opts.StructuredOutput == nil {
		drop("WithJSONMode", "true")
	}
	return warnings
}

const extraBodyUnread = "the door builds its request through a vendor SDK and has nowhere to merge them"
