package anthropic

import (
	"context"
	"slices"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic/internal/anthropicclient"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func (o *LLM) claudeReport(ctx context.Context, model string) reasoning.ClaudeReport {
	if !reasoning.IsClaude(model) || reasoning.ClaudeDocumented(model) {
		return reasoning.ClaudeReport{}
	}
	return o.models.Get(ctx, model, func(ctx context.Context) (reasoning.ClaudeReport, error) {
		info, err := o.client.GetModel(ctx, model)
		if err != nil {
			return reasoning.ClaudeReport{}, err
		}
		return claudeReportFrom(*info), nil
	})
}

func claudeReportFrom(m anthropicclient.ModelInfo) reasoning.ClaudeReport {
	info := modelInfoFrom(m)
	if !info.CapabilitiesReported {
		return reasoning.ClaudeReport{}
	}
	report := reasoning.ClaudeReport{
		Reported:          true,
		Thinking:          info.ThinkingSupported,
		Adaptive:          slices.Contains(info.ThinkingTypes, "adaptive"),
		Budget:            slices.Contains(info.ThinkingTypes, "enabled"),
		Effort:            info.EffortSupported,
		Efforts:           info.EffortLevels,
		StructuredOutputs: info.Features.StructuredOutputs,
	}
	if disabled := m.Capabilities.Thinking.Types.Disabled; disabled != nil {
		report.Disable = &disabled.Supported
	}
	return report
}

type claudeRules struct {
	model  string
	lookup func() reasoning.ClaudeReport
	asked  *reasoning.ClaudeReport
}

func (c *claudeRules) report() reasoning.ClaudeReport {
	if c.asked == nil {
		report := c.lookup()
		c.asked = &report
	}
	return *c.asked
}

func (c *claudeRules) addInherited(warn *llms.Warnings) {
	if c.asked != nil && c.asked.Reported {
		warn.AddInheritedReported(c.model)
		return
	}
	warn.AddInherited(c.model)
}

func (c *claudeRules) thinks() bool {
	return !c.report().Reported || c.report().Thinking
}

func (c *claudeRules) adaptive(preferred bool) bool {
	if !c.report().Reported {
		return reasoning.ResolveClaudeAdaptive(c.model, preferred)
	}
	switch c.report().Kind() { //nolint:exhaustive // a model that takes both types or neither keeps the caller's choice
	case reasoning.ClaudeReasoningAdaptiveOnly:
		return true
	case reasoning.ClaudeReasoningBudgetOnly:
		return false
	}
	return preferred
}

func (c *claudeRules) clampBudget(budget int) int {
	if !c.report().Reported {
		return reasoning.ClaudeClampBudget(c.model, budget)
	}
	if c.report().Budget && budget > 0 && budget < reasoning.ClaudeMinThinkingBudget {
		return reasoning.ClaudeMinThinkingBudget
	}
	return budget
}

func (c *claudeRules) clampEffort(effort string) string {
	if !c.report().Reported {
		return reasoning.ClaudeClampEffort(c.model, effort, reasoning.ProviderAnthropic)
	}
	return reasoning.ClampEffortTo(effort, c.report().Efforts)
}

func (c *claudeRules) effort(effort string, warn *llms.Warnings) string {
	if c.report().Reported && !c.report().Effort {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: c.model, Asked: effort,
			Reason: "the API reports no effort levels for this model",
		})
		return ""
	}
	return c.clampEffort(effort)
}

func (c *claudeRules) effortWithBudget() bool {
	if !c.report().Reported {
		return reasoning.ClaudeSupportsEffortWithBudget(c.model, reasoning.ProviderAnthropic)
	}
	return c.report().Effort && c.report().Budget
}

func (c *claudeRules) keepRefusal(warn *llms.Warnings, option, asked, sent string, refusal error) bool {
	return c.report().Reported || warn.KeepRefusal(c.model, option, asked, sent, refusal)
}

func (c *claudeRules) off(warn *llms.Warnings) (reasoning.OffWire, error) {
	if c.report().Reported && !c.report().Thinking {
		return reasoning.OffOmit, nil
	}
	if c.report().Reported && c.report().Disable != nil {
		if *c.report().Disable {
			return reasoning.OffDisableClaude, nil
		}
		if reasoning.InheritedOffWire(c.model, reasoning.ProviderAnthropic) == reasoning.OffBetweenToolsClaude {
			return reasoning.OffBetweenToolsClaude, nil
		}
		return reasoning.OffUnsupported, &reasoning.ErrReasoningOffUnsupported{Model: c.model}
	}
	off := reasoning.ResolveOff(c.model, reasoning.ProviderAnthropic)
	if off == reasoning.OffUnsupported {
		off = reasoning.InheritedOffWire(c.model, reasoning.ProviderAnthropic)
		if warn.KeepOffRefusal(c.model, off) {
			return reasoning.OffUnsupported, &reasoning.ErrReasoningOffUnsupported{Model: c.model}
		}
	}
	return off, nil
}

func (c *claudeRules) structuredOutput() bool {
	if !c.report().Reported {
		return reasoning.ClaudeSupportsStructuredOutput(c.model)
	}
	return c.report().StructuredOutputs
}
