package bedrockclient

import (
	"strconv"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

// A provider is listed here only where its own payload struct has the field —
// not where the vendor supports the option.
var (
	legacyCarriesTools     = map[string]bool{"anthropic": true}
	legacyCarriesTopK      = map[string]bool{"anthropic": true, "cohere": true}
	legacyCarriesStopWords = map[string]bool{"meta": false}
	legacyCarriesThinking  = map[string]bool{"anthropic": true, "nova": true}
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
	if !legacyCarriesTopK[provider] && options.TopK != nil && *options.TopK != 0 {
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
	if !legacyCarriesThinking[provider] && options.Reasoning.ResolveMode() == llms.ReasoningOn {
		reportThinkingUnsupported(warn, modelID, options.Reasoning)
	}
}

func reportThinkingUnsupported(warn *llms.Warnings, modelID string, cfg *llms.ReasoningConfig) {
	asked := "thinking"
	switch {
	case cfg.HasExplicitTokens():
		asked = strconv.Itoa(cfg.Tokens) + " tokens"
	case cfg.Effort != "":
		asked = string(cfg.Effort)
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningDrop, Option: "WithReasoning", Model: modelID,
		Asked: asked, Reason: "the door puts no thinking on the request for this model",
	})
}

func reportLegacyAnthropic(
	warn *llms.Warnings, modelID string, options llms.CallOptions, input *anthropicTextGenerationInput,
) {
	const reshaped = "the door reshaped the legacy anthropic payload for this model"

	if options.Temperature != nil {
		reportLegacyFloat(warn, "WithTemperature", modelID, reshaped, *options.Temperature, input.Temperature)
	}
	if options.TopP != nil {
		reportLegacyFloat(warn, "WithTopP", modelID, reshaped, *options.TopP, input.TopP)
	}
	if options.TopK != nil && *options.TopK != 0 && input.TopK != *options.TopK {
		reportLegacyInt(warn, "WithTopK", modelID, reshaped, *options.TopK, input.TopK)
	}
	if options.MaxTokens != nil && *options.MaxTokens > 0 && input.MaxTokens != *options.MaxTokens {
		reportLegacyInt(warn, "WithMaxTokens", modelID, reshaped, *options.MaxTokens, input.MaxTokens)
	}
	if cfg := options.Reasoning; cfg != nil && cfg.Effort != "" && cfg.Effort != llms.ReasoningNone {
		sent := ""
		if input.OutputConfig != nil {
			sent = input.OutputConfig.Effort
		}
		reportEffortClamp(warn, modelID, string(cfg.Effort), sent, input.Thinking != nil)
	}
	if input.Thinking != nil {
		reportMechanismSwap(warn, modelID, options.Reasoning, input.Thinking.Type)
	}
	if cfg := options.Reasoning; cfg != nil && cfg.HasExplicitTokens() {
		sent := 0
		if input.Thinking != nil {
			sent = input.Thinking.BudgetTokens
		}
		reportThinkingBudget(warn, modelID, cfg.Tokens, sent)
	}
	if len(options.StopWords) > 0 && len(input.StopSequences) == 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithStopWords", Model: modelID,
			Asked: strings.Join(options.StopWords, ","), Reason: reshaped,
		})
	}
}

func reportLegacyFloat(warn *llms.Warnings, option, modelID, reason string, asked, sent float64) {
	if asked == sent {
		return
	}
	render := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	if sent == 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: option, Model: modelID,
			Asked: render(asked), Reason: reason,
		})
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningSubstitute, Option: option, Model: modelID,
		Asked: render(asked), Sent: render(sent), Reason: reason,
	})
}

func reportLegacyInt(warn *llms.Warnings, option, modelID, reason string, asked, sent int) {
	if sent == 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: option, Model: modelID,
			Asked: strconv.Itoa(asked), Reason: reason,
		})
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningClamp, Option: option, Model: modelID,
		Asked: strconv.Itoa(asked), Sent: strconv.Itoa(sent), Reason: reason,
	})
}

func reportThinkingBudget(warn *llms.Warnings, modelID string, asked, sent int) {
	if asked <= 0 || sent == asked {
		return
	}
	if sent <= 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: modelID,
			Asked:  strconv.Itoa(asked) + " tokens",
			Reason: "the door puts no thinking budget on the request for this model",
		})
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningClamp, Option: "WithReasoning", Model: modelID,
		Asked: strconv.Itoa(asked) + " tokens", Sent: strconv.Itoa(sent) + " tokens",
		Reason: "the thinking budget is capped by the answer limit and by what the model records",
	})
}

func reportEffortClamp(warn *llms.Warnings, modelID, asked, sent string, thinkingSent bool) {
	switch {
	case sent == "" && thinkingSent:
		return
	case sent == "":
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: modelID,
			Asked: asked, Reason: "the door puts no thinking on the request for this model",
		})
	case !strings.EqualFold(asked, sent):
		warn.Add(llms.Warning{
			Kind: llms.WarningClamp, Option: "WithReasoning", Model: modelID,
			Asked: asked, Sent: sent,
			Reason: "the door sends only the efforts it records this model as accepting",
		})
	}
}

func reportMechanismSwap(warn *llms.Warnings, modelID string, cfg *llms.ReasoningConfig, sentType string) {
	if cfg == nil || !cfg.Adaptive || sentType == "" || sentType == "adaptive" {
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningSubstitute, Option: "WithAdaptiveReasoning", Model: modelID,
		Asked: "adaptive", Sent: sentType,
		Reason: "the door takes the thinking mechanism from the model, not from the preference",
	})
}

func reportNovaReasoning(warn *llms.Warnings, modelID string, options llms.CallOptions, effort string) {
	const cleared = "the nova request drops the sampling values at this reasoning effort"

	if cfg := options.Reasoning; cfg != nil && cfg.HasExplicitTokens() {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: modelID,
			Asked:  strconv.Itoa(cfg.Tokens) + " tokens",
			Reason: "nova takes a reasoning effort, so a token budget has nowhere to go",
		})
	}
	if !reasoning.NovaClearsInferenceConfigAt(effort) {
		return
	}
	if options.MaxTokens != nil && *options.MaxTokens > 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithMaxTokens", Model: modelID,
			Asked: strconv.Itoa(*options.MaxTokens), Reason: cleared,
		})
	}
	reportLegacyFloatDrop(warn, "WithTemperature", modelID, cleared, options.Temperature)
	reportLegacyFloatDrop(warn, "WithTopP", modelID, cleared, options.TopP)
}

func reportLegacyFloatDrop(warn *llms.Warnings, option, modelID, reason string, asked *float64) {
	if asked == nil {
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningDrop, Option: option, Model: modelID,
		Asked: strconv.FormatFloat(*asked, 'g', -1, 64), Reason: reason,
	})
}
