package mistral

import (
	"strconv"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
)

func reportMistralUnread(warn *llms.Warnings, model string, opts *llms.CallOptions) {
	const unread = "the door's request has no field for it"

	drop := func(option, asked string) {
		warn.Add(llms.Warning{
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
		{"WithTopK", opts.TopK, 0},
		{"WithMinLength", opts.MinLength, 0},
		{"WithTopLogProbs", opts.TopLogProbs, 0},
		{"WithN", opts.N, 1},
		{"WithCandidateCount", opts.CandidateCount, 1},
	} {
		if o.value != nil && *o.value != o.neutral {
			drop(o.option, strconv.Itoa(*o.value))
		}
	}
	if opts.LogProbs != nil && *opts.LogProbs {
		drop("WithLogProbs", "true")
	}
}

func reportMistralOptions(warn *llms.Warnings, model string, opts *llms.CallOptions) {
	reportMistralUnread(warn, model, opts)

	if kind, name := llms.ClassifyToolChoice(opts.ToolChoice); kind == llms.ToolChoiceNamed {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithToolChoice", Model: model,
			Asked:  name,
			Reason: "the door's tool choice is a bare string, so a named tool has no shape to travel in",
		})
	}
	if len(opts.StopWords) > 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithStopWords", Model: model,
			Asked:  strings.Join(opts.StopWords, ","),
			Reason: "the door builds no stop field",
		})
	}
	if opts.StructuredOutput != nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithStructuredOutput", Model: model,
			Asked:  opts.StructuredOutput.Name,
			Reason: "the door reads only the JSON-mode flag, never a schema",
		})
	}
	if cfg := opts.Reasoning; cfg != nil && cfg.ResolveMode() == llms.ReasoningOn {
		asked := string(cfg.GetEffort(opts.GetMaxTokens()))
		if cfg.HasExplicitTokens() {
			asked = strconv.Itoa(cfg.Tokens) + " tokens"
		}
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: model,
			Asked:  asked,
			Reason: "the door builds no reasoning field at all",
		})
	}
}
