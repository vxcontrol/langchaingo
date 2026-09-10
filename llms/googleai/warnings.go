package googleai

import (
	"strconv"
	"strings"

	"google.golang.org/genai"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func reportGoogleAIOptions(warn *llms.Warnings, model string, opts llms.CallOptions, tc *genai.ThinkingConfig) {
	if opts.N != nil && *opts.N != 1 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithN", Model: model,
			Asked:  strconv.Itoa(*opts.N),
			Reason: "the door sends candidate_count and never reads n",
		})
	}
	if opts.MinP != nil && *opts.MinP != 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithMinP", Model: model,
			Asked:  strconv.FormatFloat(*opts.MinP, 'g', -1, 64),
			Reason: "the door's generation config has no min-p field to set",
		})
	}
	if opts.RepetitionPenalty != nil && *opts.RepetitionPenalty != 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithRepetitionPenalty", Model: model,
			Asked:  strconv.FormatFloat(*opts.RepetitionPenalty, 'g', -1, 64),
			Reason: "the door's generation config has no repetition-penalty field to set",
		})
	}
	if opts.LogProbs != nil && *opts.LogProbs {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithLogProbs", Model: model,
			Asked:  "true",
			Reason: "the door never sets logprobs on the generation config it builds",
		})
	}
	if opts.TopLogProbs != nil && *opts.TopLogProbs != 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTopLogProbs", Model: model,
			Asked:  strconv.Itoa(*opts.TopLogProbs),
			Reason: "the door never sets logprobs on the generation config it builds",
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
		sent := string(genai.ThinkingLevelHigh)
		asked := string(cfg.GetEffort(opts.GetMaxTokens()))
		if cfg.HasExplicitTokens() {
			asked = strconv.Itoa(cfg.Tokens) + " tokens"
		}
		if !strings.EqualFold(asked, sent) {
			warn.Add(llms.Warning{
				Kind: llms.WarningSubstitute, Option: "WithReasoning", Model: model,
				Asked: asked, Sent: sent,
				Reason: "the door drives this family by thinking level and sends its top level whatever was asked",
			})
		}
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
