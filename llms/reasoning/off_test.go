package reasoning

import "testing"

func TestResolveOff(t *testing.T) {
	t.Parallel()
	cases := []struct {
		model string
		p     Provider
		want  OffWire
	}{
		{"claude-fable-5", ProviderAnthropic, OffUnsupported},
		{"us.anthropic.claude-mythos-5", ProviderBedrock, OffUnsupported},
		{"claude-sonnet-5", ProviderAnthropic, OffDisableClaude},
		{"us.anthropic.claude-sonnet-5", ProviderBedrock, OffDisableClaude},
		{"anthropic/claude-sonnet-5", ProviderOpenAI, OffDisableThinkingObject},
		{"anthropic/claude-opus-5", ProviderOpenAI, OffDisableThinkingObject},
		{"anthropic/claude-fable-5", ProviderOpenAI, OffUnsupported},
		{"anthropic/claude-opus-4-8", ProviderOpenAI, OffOmit},
		{"claude-opus-5", ProviderAnthropic, OffDisableClaude},
		{"us.anthropic.claude-opus-5", ProviderBedrock, OffDisableClaude},
		{"claude-opus-4-8", ProviderAnthropic, OffOmit},  // off by default
		{"claude-sonnet-4-5", ProviderBedrock, OffOmit},  // off by default
		{"claude-3-5-haiku", ProviderAnthropic, OffOmit}, // no thinking
		{"gemini-2.5-flash", ProviderGoogleAI, OffZeroBudget},
		{"gemini-2.5-flash-lite", ProviderGoogleAI, OffZeroBudget},
		{"gemini-2.5-pro", ProviderGoogleAI, OffUnsupported}, // Pro cannot disable thinking
		{"gemini-3.1-pro", ProviderGoogleAI, OffUnsupported},
		{"gemini-3.1-pro-preview", ProviderGoogleAI, OffUnsupported},
		{"gemini-3.1-pro-preview-customtools", ProviderGoogleAI, OffUnsupported},
		{"gemini-3.1-flash-lite", ProviderGoogleAI, OffMinimalLevel},
		{"gemini-3-flash", ProviderGoogleAI, OffMinimalLevel},
		{"gemini-3-flash-preview", ProviderGoogleAI, OffMinimalLevel},
		{"gemini-3.5-flash", ProviderGoogleAI, OffMinimalLevel},
		{"gemini-3.5-flash-lite", ProviderGoogleAI, OffMinimalLevel},
		{"models/gemini-3.5-flash", ProviderGoogleAI, OffMinimalLevel},
		{"gemini-3.6-flash", ProviderGoogleAI, OffMinimalLevel},
		{"gemini-3.7-flash", ProviderGoogleAI, OffUnsupported},
		{"gemini-3.8-flash", ProviderGoogleAI, OffUnsupported},
		{"gemini-3.5-pro", ProviderGoogleAI, OffUnsupported},
		{"gemma-4-31b-it", ProviderGoogleAI, OffMinimalLevel},
		{"gemma-4-26b-a4b-it", ProviderGoogleAI, OffMinimalLevel},
		// Pre-thinking Gemini/Gemma never think: omit rather than send budget:0.
		{"gemini-2.0-flash", ProviderGoogleAI, OffOmit},
		{"gemini-1.5-pro", ProviderGoogleAI, OffOmit},
		{"gemma-3-27b-it", ProviderGoogleAI, OffOmit},
		{"gpt-5.5", ProviderOpenAI, OffEffortNone},
		{"o3-mini", ProviderOpenAI, OffUnsupported},
		{"o1-preview", ProviderOpenAI, OffUnsupported},
		{"gpt-5-pro", ProviderOpenAI, OffUnsupported},          // accepts only high, cannot disable
		{"gpt-4.1", ProviderOpenAI, OffOmit},                   // non-reasoning
		{"some-random-model", ProviderGoogleAI, OffZeroBudget}, // optimistic google
	}
	for _, tc := range cases {
		if got := ResolveOff(tc.model, tc.p); got != tc.want {
			t.Errorf("ResolveOff(%q,%d) = %d, want %d", tc.model, tc.p, got, tc.want)
		}
	}
}

func TestVendorsThatDisableWithAThinkingObject(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"glm-4.5", "glm-4.6", "glm-4.7", "glm-5", "glm-5-turbo", "glm-5.1", "glm-5.2",
		"minimax-m3", "deepseek-v4-flash", "deepseek-v4-pro",
		"zai/glm-5.2", "deepseek/deepseek-v4-flash",
	} {
		if got := ResolveOff(model, ProviderOpenAI); got != OffDisableThinkingObject {
			t.Errorf("ResolveOff(%q) = %v, want the thinking object: the effort token is either "+
				"ignored by the vendor or cut before it", model, got)
		}
	}

	for model, want := range map[string]OffWire{
		"glm-5.3":    OffUnsupported,
		"minimax-m2": OffUnsupported,
		"kimi-k3":    OffUnsupported,
	} {
		if got := ResolveOff(model, ProviderOpenAI); got != want {
			t.Errorf("ResolveOff(%q) = %v, want %v", model, got, want)
		}
	}
}

func TestDashScopeGuestsSpellOffWithTheDashScopeFlag(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"dashscope/glm-5.2", "dashscope/glm-5.1", "dashscope/glm-5", "dashscope/glm-4.7",
		"dashscope/glm-4.6", "dashscope/glm-4.5",
		"dashscope/deepseek-v4-pro", "dashscope/deepseek-v4-flash", "dashscope/deepseek-v4-flash-0731",
		"dashscope/kimi-k2.6",
	} {
		if got := ResolveOff(model, ProviderOpenAI); got != OffDisableDashScope {
			t.Errorf("ResolveOff(%q, openai) = %v, want OffDisableDashScope", model, got)
		}
	}
}

func TestGLMServedByMistralSpellsOffByOmission(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"zai-glm-5-2", "mistral/zai-glm-5-2", "glm-5-2", "mistral/glm-5-2",
	} {
		if got := ResolveOff(model, ProviderOpenAI); got != OffOmit {
			t.Errorf("ResolveOff(%q, openai) = %v, want OffOmit", model, got)
		}
	}
}

func TestAlwaysThinkingFamiliesAreRefusedOnBedrockToo(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model string
		want  OffWire
	}{
		{"kimi-k3", OffUnsupported},
		{"moonshot.kimi-k2-thinking", OffUnsupported},
		{"us.deepseek.r1-v1:0", OffUnsupported},
		{"openai.gpt-oss-120b-1:0", OffUnsupported},
		{"us.xai.grok-4.6", OffUnsupported},
		{"glm-5.3", OffUnsupported},
		{"us.amazon.nova-2-lite-v1:0", OffOmit},
		{"xai.grok-4.3", OffOmit},
	} {
		if got := ResolveOff(tc.model, ProviderBedrock); got != tc.want {
			t.Errorf("ResolveOff(%q, ProviderBedrock) = %v, want %v", tc.model, got, tc.want)
		}
	}
}
