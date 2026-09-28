package googleai

import (
	"strconv"
	"strings"

	"google.golang.org/genai"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func reportGoogleAIOptions(warn *llms.Warnings, model string, opts llms.CallOptions, tc *genai.ThinkingConfig) {
	warn.AddUnreadExtraBody(model, opts, extraBodyUnread)

	if opts.N != nil && *opts.N != 1 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithN", Model: model,
			Asked:  strconv.Itoa(*opts.N),
			Reason: "the door sends candidate_count and never reads n",
		})
	}
	warn.AddUnreadOptions(model, opts, "the door's generation config has no field to set",
		"WithTopK", "WithCandidateCount", "WithSeed", "WithN",
		"WithFrequencyPenalty", "WithPresencePenalty", "WithResponseMIMEType", "WithJSONMode")
	reportGoogleAIThinking(warn, model, opts, tc)
}

func reportGoogleAIThinking(warn *llms.Warnings, model string, opts llms.CallOptions, tc *genai.ThinkingConfig) {
	cfg := opts.Reasoning
	if cfg == nil {
		return
	}
	if cfg.ResolveMode() == llms.ReasoningOff {
		reportGoogleAIDisableFloor(warn, model, tc)
		return
	}
	if cfg.ResolveMode() != llms.ReasoningOn {
		return
	}
	if !reasoning.GeminiSupportsThinking(model) {
		asked := string(cfg.GetEffort(opts.GetMaxTokens()))
		if cfg.HasExplicitTokens() {
			asked = strconv.Itoa(cfg.Tokens) + " tokens"
		}
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: model,
			Asked: asked, Reason: "this model does not think, so the door sends no thinking config",
		})
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

	if tc != nil && tc.ThinkingLevel != "" {
		asked := string(cfg.GetEffort(opts.GetMaxTokens()))
		if cfg.Effort != "" && !strings.EqualFold(asked, string(tc.ThinkingLevel)) {
			warn.Add(llms.Warning{
				Kind: llms.WarningSubstitute, Option: "WithReasoning", Model: model,
				Asked: asked, Sent: string(tc.ThinkingLevel),
				Reason: "the door sends only the thinking levels it records this model as accepting",
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
			Reason: "the thinking budget is held inside the model's range and below two thirds of the answer limit",
		})
	}
}

func reportGoogleAIDisableFloor(warn *llms.Warnings, model string, tc *genai.ThinkingConfig) {
	if tc == nil || tc.ThinkingLevel == "" || reasoning.GeminiTogglesThinkingByLevel(model) {
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningSubstitute, Option: "WithReasoningDisabled", Model: model,
		Asked: "off", Sent: strings.ToLower(string(tc.ThinkingLevel)),
		Reason: "this model has no off switch, only a lowest thinking level",
	})
}

const extraBodyUnread = "the door builds its request through a vendor SDK and has nowhere to merge them"
