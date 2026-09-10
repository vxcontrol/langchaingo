package bedrockclient

import (
	"strconv"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
)

// A provider is listed here only where its own payload struct has the field —
// not where the vendor supports the option.
var (
	legacyCarriesTools     = map[string]bool{"anthropic": true}
	legacyCarriesTopK      = map[string]bool{"anthropic": true, "cohere": true}
	legacyCarriesStopWords = map[string]bool{"meta": false}
)

func reportLegacyOptions(warn *llms.Warnings, provider, modelID string, options llms.CallOptions) {
	reason := "the legacy " + provider + " payload has no field for it"

	if !legacyCarriesTools[provider] {
		if len(options.Tools) > 0 {
			warn.Add(llms.Warning{
				Kind: llms.WarningDrop, Option: "WithTools", Model: modelID,
				Asked: strconv.Itoa(len(options.Tools)) + " tools", Reason: reason,
			})
		}
		if kind, name := llms.ClassifyToolChoice(options.ToolChoice); kind != llms.ToolChoiceUnset {
			asked := name
			if asked == "" {
				asked = "kind " + strconv.Itoa(int(kind))
			}
			warn.Add(llms.Warning{
				Kind: llms.WarningDrop, Option: "WithToolChoice", Model: modelID,
				Asked: asked, Reason: reason,
			})
		}
	}
	if !legacyCarriesTopK[provider] && options.TopK != nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTopK", Model: modelID,
			Asked: strconv.Itoa(*options.TopK), Reason: reason,
		})
	}
	if carries, known := legacyCarriesStopWords[provider]; known && !carries && len(options.StopWords) > 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithStopWords", Model: modelID,
			Asked: strings.Join(options.StopWords, ","), Reason: reason,
		})
	}
}
