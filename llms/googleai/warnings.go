package googleai

import (
	"strconv"

	"google.golang.org/genai"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func reportGoogleAIOptions(warn *llms.Warnings, model string, opts llms.CallOptions, tc *genai.ThinkingConfig) {
	if opts.N != nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithN", Model: model,
			Asked:  strconv.Itoa(*opts.N),
			Reason: "the door sends candidate_count and never reads n",
		})
	}
	if opts.LogProbs != nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithLogProbs", Model: model,
			Asked:  strconv.FormatBool(*opts.LogProbs),
			Reason: "the door's generation config carries no logprobs field",
		})
	}
	if opts.TopLogProbs != nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTopLogProbs", Model: model,
			Asked:  strconv.Itoa(*opts.TopLogProbs),
			Reason: "the door's generation config carries no logprobs field",
		})
	}
	reportGoogleAIThinking(warn, model, opts, tc)
}

func reportGoogleAIThinking(warn *llms.Warnings, model string, opts llms.CallOptions, tc *genai.ThinkingConfig) {
	cfg := opts.Reasoning
	if cfg == nil || cfg.ResolveMode() != llms.ReasoningOn {
		return
	}

	if reasoning.GeminiTogglesThinkingByLevel(model) {
		asked := string(cfg.GetEffort(opts.GetMaxTokens()))
		if cfg.HasExplicitTokens() {
			asked = strconv.Itoa(cfg.Tokens) + " tokens"
		}
		warn.Add(llms.Warning{
			Kind: llms.WarningSubstitute, Option: "WithReasoning", Model: model,
			Asked: asked, Sent: string(genai.ThinkingLevelHigh),
			Reason: "the door drives this family by thinking level and sends its top level whatever was asked",
		})
		return
	}

	if !cfg.HasExplicitTokens() || tc == nil || tc.ThinkingBudget == nil {
		return
	}
	if sent := int(*tc.ThinkingBudget); sent != cfg.Tokens {
		warn.Add(llms.Warning{
			Kind: llms.WarningClamp, Option: "WithReasoning", Model: model,
			Asked: strconv.Itoa(cfg.Tokens), Sent: strconv.Itoa(sent),
			Reason: "the thinking budget is capped at two thirds of the answer limit",
		})
	}
}
