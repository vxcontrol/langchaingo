package llms

import (
	"errors"
	"testing"

	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestValidateReasoningRefusesOnlyWhatNoDoorAccepts(t *testing.T) {
	t.Parallel()

	for _, effort := range []ReasoningEffort{
		ReasoningNone, ReasoningMinimal, ReasoningLow,
		ReasoningMedium, ReasoningHigh, ReasoningXHigh, ReasoningMax,
	} {
		opts := CallOptions{Reasoning: &ReasoningConfig{Effort: effort}}
		if err := opts.ValidateReasoning(); err != nil {
			t.Errorf("effort %q is offered by at least one door, got %v", effort, err)
		}
	}

	opts := CallOptions{Reasoning: &ReasoningConfig{Effort: ReasoningEffort("banana")}}
	err := opts.ValidateReasoning()
	var typed *Error
	if !errors.As(err, &typed) || typed.Code != ErrCodeInvalidRequest {
		t.Fatalf("an effort no door accepts must be refused, got %v", err)
	}

	var absent CallOptions
	if err := absent.ValidateReasoning(); err != nil {
		t.Errorf("a call that asked for no reasoning is valid, got %v", err)
	}
}

func TestTheCeilingRisesOnlyWhenTheBudgetWouldNotFit(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name              string
		budget, maxTokens int
		want              int
	}{
		{"a budget that does not fit raises the ceiling", 4096, 1024, 8192},
		{"a budget at the ceiling still raises it", 1024, 1024, 2048},
		{"a budget that fits leaves the caller's ceiling alone", 40000, 64000, 64000},
		{"no budget leaves the ceiling alone", 0, 64000, 64000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := reasoning.ClaudeMaxTokensForBudget(tc.budget, tc.maxTokens); got != tc.want {
				t.Fatalf("ceiling = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestTheEffortNoneReadsAsReasoningOff(t *testing.T) {
	t.Parallel()

	shared := &ReasoningConfig{Effort: ReasoningEffort("none")}
	opts := CallOptions{Reasoning: shared}
	if err := opts.ValidateReasoning(); err != nil {
		t.Fatalf("OpenAI documents none as an effort, got %v", err)
	}
	if got := opts.Reasoning.ResolveMode(); got != ReasoningOff {
		t.Errorf("mode = %v, want ReasoningOff", got)
	}
	if shared.Effort != ReasoningEffort("none") {
		t.Errorf("the caller's config must stay as written, got %+v", shared)
	}

	adaptive := CallOptions{Reasoning: &ReasoningConfig{Effort: ReasoningEffort("none"), Adaptive: true}}
	if err := adaptive.ValidateReasoning(); err != nil {
		t.Fatalf("WithAdaptiveReasoning(none) is valid, got %v", err)
	}
	if got := adaptive.Reasoning.ResolveMode(); got != ReasoningOff || adaptive.Reasoning.DelegatesDepth() {
		t.Errorf("none turns reasoning off even beside adaptive, got %+v", adaptive.Reasoning)
	}

	budget := CallOptions{Reasoning: &ReasoningConfig{Effort: ReasoningEffort("none"), Tokens: 2048}}
	if err := budget.ValidateReasoning(); err != nil {
		t.Fatalf("none beside a budget leaves the depth to the budget, got %v", err)
	}
	if budget.Reasoning.ResolveMode() != ReasoningOn || budget.Reasoning.Effort != ReasoningNone {
		t.Errorf("none with a budget must think on the budget alone, got %+v", budget.Reasoning)
	}
}
