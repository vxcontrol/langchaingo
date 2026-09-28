package reasoning

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestProbeR2Diff(t *testing.T) {
	body, err := os.ReadFile("/tmp/claude-0/-home-user-langchaingo/190be914-a4ca-51a0-89e2-3fbdbb6fc0c9/scratchpad/probes/r2diff/names.txt")
	if err != nil {
		t.Fatal(err)
	}
	out := &strings.Builder{}
	for _, m := range strings.Split(string(body), "\n") {
		if m == "" {
			continue
		}
		caps := OpenAIReasoningCapsFor(m)
		fmt.Fprintf(out, "%s\tisR=%v\tkind=%d\talways=%v\tdefOn=%v\tmutex=%v\tSO=%v\tadaptT=%v\tadaptF=%v\tpre=%v\trejS=%v\tcaps=%v/%v/%v\tgThink=%v\tgLvl=%v\tgDis=%v\toffA=%s\toffB=%s\toffO=%s\toffG=%s\toffU=%s\n",
			m, IsReasoningModel(m), ClaudeReasoningKindFor(m), ClaudeThinkingAlwaysOn(m), ClaudeThinkingDefaultsOn(m),
			ClaudeMutuallyExclusiveSampling(m), ClaudeSupportsStructuredOutput(m), ResolveClaudeAdaptive(m, true), ResolveClaudeAdaptive(m, false),
			ClaudePredatesAdaptive(m), ClaudeRejectsSampling(m), caps.Known, caps.CanDisable, caps.Efforts,
			GeminiSupportsThinking(m), GeminiUsesThinkingLevel(m), GeminiCanDisable(m),
			offName(ResolveOff(m, ProviderAnthropic)), offName(ResolveOff(m, ProviderBedrock)), offName(ResolveOff(m, ProviderOpenAI)), offName(ResolveOff(m, ProviderGoogleAI)), offName(ResolveOff(m, ProviderUnknown)))
	}
	if err := os.WriteFile(os.Getenv("PROBE_OUT"), []byte(out.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

func offName(w OffWire) string {
	switch w {
	case OffOmit:
		return "omit"
	case OffDisableClaude:
		return "claudeOff"
	case OffZeroBudget:
		return "zeroBudget"
	case OffEffortNone:
		return "effortNone"
	case OffUnsupported:
		return "UNSUPPORTED"
	}
	return fmt.Sprintf("other%d", int(w))
}
