package reasoning

import "testing"

func TestProbeV3Effort(t *testing.T) {
	for _, m := range []string{"us.anthropic.claude-fable-5-1", "us.anthropic.claude-mythos-5-1", "us.anthropic.claude-opus-5-5", "anthropic.claude-fable-5", "us.anthropic.claude-sonnet-5", "us.anthropic.claude-opus-4-7"} {
		t.Logf("%s bedrock=%v anthropic=%v clamp(xhigh)=%s clamp(max)=%s", m, ClaudeEffortsFor(m, ProviderBedrock), ClaudeEffortsFor(m, ProviderAnthropic), ClaudeClampEffort(m, "xhigh", ProviderBedrock), ClaudeClampEffort(m, "max", ProviderBedrock))
	}
}
