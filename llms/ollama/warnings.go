package ollama

import (
	"strconv"

	"github.com/ollama/ollama/api"

	"github.com/vxcontrol/langchaingo/llms"
)

func reportOllamaOptions(warn *llms.Warnings, model string, opts llms.CallOptions) {
	const unread = "the door builds no field for it"

	warn.AddUnreadExtraBody(model, opts, extraBodyUnread)

	if opts.Reasoning.DelegatesDepth() {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithAdaptiveReasoning", Model: model,
			Asked:  "adaptive",
			Reason: "the door has no field that asks the model to choose its own depth",
		})
	}

	warn.AddUnreadOptions(model, opts, unread,
		"WithRepetitionPenalty", "WithFrequencyPenalty", "WithPresencePenalty",
		"WithTopK", "WithSeed", "WithJSONMode", "WithMinP")
	if opts.MinP != nil && *opts.MinP != 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithMinP", Model: model,
			Asked:  strconv.FormatFloat(*opts.MinP, 'g', -1, 64),
			Reason: "the door sends only the min_p set on the client",
		})
	}
	if kind, name := llms.ClassifyToolChoice(opts.ToolChoice); kind != llms.ToolChoiceUnset &&
		kind != llms.ToolChoiceAuto {
		asked := name
		if asked == "" {
			asked = kind.String()
		}
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithToolChoice", Model: model,
			Asked: asked, Reason: unread,
		})
	}
	reportOllamaThinking(warn, model, opts)
}

func reportOllamaThinking(warn *llms.Warnings, model string, opts llms.CallOptions) {
	cfg := opts.Reasoning
	if cfg == nil || cfg.ResolveMode() != llms.ReasoningOn {
		return
	}

	asked := cfg.GetEffort(opts.GetMaxTokens())
	effort := string(asked)
	switch {
	case takesOnlyGPTOSSLevels(model):
		if sent := gptOSSLevel(asked); sent != effort {
			warn.Add(llms.Warning{
				Kind: llms.WarningSubstitute, Option: "WithReasoning", Model: model,
				Asked: effort, Sent: sent,
				Reason: "this model takes only low, medium or high, and ignores anything else",
			})
		}
	default:
		if level := (&api.ThinkValue{Value: effort}); !level.IsValid() {
			warn.Add(llms.Warning{
				Kind: llms.WarningSubstitute, Option: "WithReasoning", Model: model,
				Asked: effort, Sent: "true",
				Reason: "ollama takes a think level from a closed set and this effort is not in it",
			})
		}
	}
	if cfg.HasExplicitTokens() {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: model,
			Asked:  strconv.Itoa(cfg.Tokens),
			Reason: "ollama expresses thinking as a level, so a token budget has nowhere to go",
		})
	}
}

const extraBodyUnread = "the door builds its request through a vendor SDK and has nowhere to merge them"

const ollamaCloudStructuredOutputFallbackReason = "Ollama Cloud does not support structured outputs, " +
	"so the schema travels in the prompt and the answer is validated locally"

// reportOllamaCloudFormat reports the format a cloud-served model cannot take.
// emulated marks a structured-output call that WithCloudStructuredOutputFallback
// turned into a prompt instruction.
func reportOllamaCloudFormat(warn *llms.Warnings, model string, opts llms.CallOptions, clientFormat string, emulated bool) {
	switch {
	case emulated:
		asked := opts.StructuredOutput.Name
		if asked == "" {
			asked = "a JSON Schema"
		}
		warn.Add(llms.Warning{
			Kind: llms.WarningSubstitute, Option: "WithStructuredOutput", Model: model,
			Asked: asked, Sent: "a prompt instruction", Reason: ollamaCloudStructuredOutputFallbackReason,
		})
	case opts.JSONMode:
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithJSONMode", Model: model,
			Asked: "true", Reason: ollamaCloudFormatReason,
		})
	case clientFormat != "":
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithFormat", Model: model,
			Asked: clientFormat, Reason: ollamaCloudFormatReason,
		})
	}
}
