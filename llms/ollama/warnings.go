package ollama

import (
	"fmt"
	"strconv"

	"github.com/ollama/ollama/api"

	"github.com/vxcontrol/langchaingo/llms"
)

func reportOllamaOptions(warn *llms.Warnings, model string, opts llms.CallOptions) {
	const unread = "the door builds no field for it"

	if opts.MinP != nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithMinP", Model: model,
			Asked:  strconv.FormatFloat(*opts.MinP, 'g', -1, 64),
			Reason: "the door sends only the min_p set on the client",
		})
	}
	if opts.LogProbs != nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithLogProbs", Model: model,
			Asked: strconv.FormatBool(*opts.LogProbs), Reason: unread,
		})
	}
	if opts.TopLogProbs != nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTopLogProbs", Model: model,
			Asked: strconv.Itoa(*opts.TopLogProbs), Reason: unread,
		})
	}
	if opts.N != nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithN", Model: model,
			Asked: strconv.Itoa(*opts.N), Reason: unread,
		})
	}
	if opts.CandidateCount != nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithCandidateCount", Model: model,
			Asked: strconv.Itoa(*opts.CandidateCount), Reason: unread,
		})
	}
	if kind, name := llms.ClassifyToolChoice(opts.ToolChoice); kind != llms.ToolChoiceUnset {
		asked := name
		if asked == "" {
			asked = fmt.Sprintf("kind %d", kind)
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

	effort := string(cfg.GetEffort(opts.GetMaxTokens()))
	if level := (&api.ThinkValue{Value: effort}); !level.IsValid() {
		warn.Add(llms.Warning{
			Kind: llms.WarningSubstitute, Option: "WithReasoning", Model: model,
			Asked: effort, Sent: "true",
			Reason: "ollama takes a think level from a closed set and this effort is not in it",
		})
	}
	if cfg.HasExplicitTokens() {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoningTokens", Model: model,
			Asked:  strconv.Itoa(cfg.Tokens),
			Reason: "ollama expresses thinking as a level, so a token budget has nowhere to go",
		})
	}
}
