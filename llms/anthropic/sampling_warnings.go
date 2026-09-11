package anthropic

import (
	"strconv"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic/internal/anthropicclient"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func reportAnthropicUnread(warn *llms.Warnings, model string, opts llms.CallOptions) {
	const unread = "the door builds no field for it"

	warn.AddUnreadExtraBody(model, opts, extraBodyUnread)
	warn.AddUnreadOptions(model, opts, unread, "WithTopK")
}

func reportAnthropicEffort(
	warn *llms.Warnings, model string, opts llms.CallOptions,
	thinking *anthropicclient.ThinkingPayload, outputConfig *anthropicclient.OutputConfig,
) {
	cfg := opts.Reasoning
	if cfg == nil || cfg.Effort == "" || cfg.Effort == llms.ReasoningNone {
		return
	}
	asked := string(cfg.Effort)
	sent := ""
	if outputConfig != nil {
		sent = outputConfig.Effort
	}
	switch {
	case sent == "" && thinking != nil:
		return
	case sent == "":
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: model,
			Asked: asked, Reason: "the door puts no thinking on the request for this model",
		})
	case !strings.EqualFold(asked, sent):
		warn.Add(llms.Warning{
			Kind: llms.WarningClamp, Option: "WithReasoning", Model: model,
			Asked: asked, Sent: sent,
			Reason: "the door sends only the efforts it records this model as accepting",
		})
	}
}

func reportAnthropicMechanism(warn *llms.Warnings, model string, opts llms.CallOptions, thinking *anthropicclient.ThinkingPayload) {
	cfg := opts.Reasoning
	if cfg == nil || !cfg.Adaptive || thinking == nil || thinking.Type == "adaptive" {
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningSubstitute, Option: "WithAdaptiveReasoning", Model: model,
		Asked: "adaptive", Sent: thinking.Type,
		Reason: "the door takes the thinking mechanism from the model, not from the preference",
	})
}

func reportAnthropicBudget(warn *llms.Warnings, model string, opts llms.CallOptions, thinking *anthropicclient.ThinkingPayload) {
	cfg := opts.Reasoning
	if cfg == nil || !cfg.HasExplicitTokens() {
		return
	}
	sent := 0
	if thinking != nil {
		sent = thinking.Budget
	}
	if sent == cfg.Tokens {
		return
	}
	if sent <= 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: model,
			Asked:  strconv.Itoa(cfg.Tokens) + " tokens",
			Reason: "the door sends no thinking budget on this model",
		})
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningClamp, Option: "WithReasoning", Model: model,
		Asked: strconv.Itoa(cfg.Tokens) + " tokens", Sent: strconv.Itoa(sent) + " tokens",
		Reason: "the thinking budget is capped by the answer limit and by what the model records",
	})
}

func reportAnthropicSampling(
	warn *llms.Warnings, model string, opts llms.CallOptions,
	thinking *anthropicclient.ThinkingPayload, outputConfig *anthropicclient.OutputConfig,
	temperature, topP *float64, topK *int, maxTokens int,
) {
	reportAnthropicUnread(warn, model, opts)
	reportAnthropicBudget(warn, model, opts, thinking)
	reportAnthropicMechanism(warn, model, opts, thinking)
	reportAnthropicEffort(warn, model, opts, thinking, outputConfig)

	reason := anthropicSamplingReason(model, thinking)
	warn.AddFloatChange("WithTemperature", model, reason, opts.Temperature, temperature)
	warn.AddFloatChange("WithTopP", model, reason, opts.TopP, topP)
	if opts.TopK != nil && *opts.TopK != 0 {
		warn.AddIntChange("WithTopK", model, reason, opts.TopK, topK)
	}

	if asked := opts.GetMaxTokens(); asked > 0 && maxTokens != asked {
		warn.Add(llms.Warning{
			Kind: llms.WarningClamp, Option: "WithMaxTokens", Model: model,
			Asked: strconv.Itoa(asked), Sent: strconv.Itoa(maxTokens),
			Reason: "the answer limit was raised to leave room for the thinking budget",
		})
	}
}

func anthropicSamplingReason(model string, thinking *anthropicclient.ThinkingPayload) string {
	switch {
	case reasoning.ClaudeRejectsSampling(model):
		return "the model rejects sampling parameters"
	case thinking != nil && thinking.Type == "adaptive":
		return "adaptive thinking takes no sampling parameters"
	case thinking != nil && thinking.Type == "enabled":
		return "the model refuses sampling while thinking"
	default:
		return "the model does not accept this combination of sampling parameters"
	}
}

const extraBodyUnread = "the door builds its request through a vendor SDK and has nowhere to merge them"

func reportAnthropicCompletions(warn *llms.Warnings, model string, opts llms.CallOptions) {
	const unread = "the legacy text-completions request has no field for it"

	warn.AddUnreadExtraBody(model, opts, extraBodyUnread)
	warn.AddUnreadOptions(model, opts, unread)

	drop := func(option, asked string) {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: option, Model: model,
			Asked: asked, Reason: unread,
		})
	}
	if len(opts.Tools) > 0 {
		drop("WithTools", strconv.Itoa(len(opts.Tools))+" tools")
	}
	if kind, name := llms.ClassifyToolChoice(opts.ToolChoice); kind != llms.ToolChoiceUnset {
		asked := name
		if asked == "" {
			asked = kind.String()
		}
		drop("WithToolChoice", asked)
	}
	if opts.StructuredOutput != nil {
		drop("WithStructuredOutput", opts.StructuredOutput.Name)
	}
	if cfg := opts.Reasoning; cfg != nil && cfg.ResolveMode() != llms.ReasoningDefault {
		asked := string(cfg.GetEffort(opts.GetMaxTokens()))
		if cfg.HasExplicitTokens() {
			asked = strconv.Itoa(cfg.Tokens) + " tokens"
		}
		drop("WithReasoning", asked)
	}
}
