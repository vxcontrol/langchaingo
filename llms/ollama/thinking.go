package ollama

import (
	"context"
	"fmt"
	"slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type modelThinking struct {
	reported   bool
	thinks     bool
	descriptor *model.Thinking
}

func (o *LLM) thinkingOf(ctx context.Context, name string) (modelThinking, error) {
	if cached, ok := o.thinking.Load(name); ok {
		return cached.(modelThinking), nil
	}
	resp, err := o.client.Show(ctx, &api.ShowRequest{Model: name})
	if err != nil {
		return modelThinking{}, fmt.Errorf("ollama: show model %q: %w", name, err)
	}
	info := modelThinking{
		reported:   len(resp.Capabilities) > 0,
		thinks:     slices.Contains(resp.Capabilities, model.CapabilityThinking),
		descriptor: resp.Thinking,
	}
	o.thinking.Store(name, info)
	return info, nil
}

func (o *LLM) thinkingFor(ctx context.Context, name string, opts llms.CallOptions) (modelThinking, error) {
	mode := opts.Reasoning.ResolveMode()
	if mode != llms.ReasoningOn && mode != llms.ReasoningOff {
		return modelThinking{}, nil
	}
	if mode == llms.ReasoningOff && reasoning.ResolveOff(name, reasoning.ProviderOllama) == reasoning.OffUnsupported {
		return modelThinking{}, &reasoning.ErrReasoningOffUnsupported{Model: name}
	}

	info, err := o.thinkingOf(ctx, name)
	if err != nil {
		return modelThinking{}, err
	}
	if mode == llms.ReasoningOff && info.descriptor.Valid() && !info.descriptor.Supports(false) {
		return modelThinking{}, &reasoning.ErrReasoningOffUnsupported{Model: name}
	}
	return info, nil
}

func chooseThink(name string, opts llms.CallOptions, info modelThinking, warn *llms.Warnings) *api.ThinkValue {
	switch mode := opts.Reasoning.ResolveMode(); {
	case mode == llms.ReasoningOff:
		return &api.ThinkValue{Value: false}
	case mode != llms.ReasoningOn, opts.Reasoning.DelegatesDepth():
		return nil
	case info.reported && !info.thinks:
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithReasoning", Model: name,
			Asked:  string(opts.Reasoning.GetEffort(opts.GetMaxTokens())),
			Reason: "the server lists no thinking capability for this model and refuses think for it",
		})
		return nil
	case !info.descriptor.Valid():
		reportOllamaThinking(warn, name, opts)
		return thinkByName(name, opts)
	}

	asked := string(opts.Reasoning.GetEffort(opts.GetMaxTokens()))
	sent := describedThink(info.descriptor, asked)
	if sent != asked {
		warn.Add(llms.Warning{
			Kind: llms.WarningSubstitute, Option: "WithReasoning", Model: name,
			Asked: asked, Sent: fmt.Sprint(sent),
			Reason: "the server lists the think values this model takes and this effort is not among them",
		})
	}
	reportOllamaBudget(warn, name, opts, fmt.Sprint(sent))
	return &api.ThinkValue{Value: sent}
}

var effortRank = map[string]int{"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}

func describedThink(descriptor *model.Thinking, asked string) any {
	if descriptor.Supports(asked) {
		return asked
	}
	want, ranked := effortRank[asked]
	best, bestGap := "", 0
	for _, value := range descriptor.Values {
		level, ok := value.(string)
		rank, known := effortRank[level]
		if !ok || !known || !ranked {
			continue
		}
		gap := max(rank-want, want-rank)
		if best == "" || gap < bestGap || (gap == bestGap && rank < effortRank[best]) {
			best, bestGap = level, gap
		}
	}
	switch {
	case best != "":
		return best
	case descriptor.Supports(true):
		return true
	default:
		return descriptor.Default
	}
}
