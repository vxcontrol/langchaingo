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
		"WithTopK", "WithSeed", "WithJSONMode")
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
