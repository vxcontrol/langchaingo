package mistral

import (
	"strconv"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
)

func reportMistralOptions(warn *llms.Warnings, model string, opts *llms.CallOptions) {
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
