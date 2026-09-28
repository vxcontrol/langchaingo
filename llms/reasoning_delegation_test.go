package llms

import "testing"

func TestAnExplicitOffIsNotADelegationOfDepth(t *testing.T) {
	t.Parallel()

	cfg := &ReasoningConfig{Mode: ReasoningOff, Adaptive: true}
	if cfg.DelegatesDepth() {
		t.Error("a caller who switched thinking off did not hand the depth to the vendor")
	}
	if cfg.ResolveMode() != ReasoningOff {
		t.Errorf("ResolveMode() = %v, want %v", cfg.ResolveMode(), ReasoningOff)
	}
}

func TestAdaptiveWithoutAModeStillDelegatesDepth(t *testing.T) {
	t.Parallel()

	if !(&ReasoningConfig{Adaptive: true}).DelegatesDepth() {
		t.Error("adaptive with no effort and no budget leaves the depth to the vendor")
	}
	if !(&ReasoningConfig{Mode: ReasoningOn, Adaptive: true}).DelegatesDepth() {
		t.Error("an explicit on with the adaptive flag still delegates the depth")
	}
}
