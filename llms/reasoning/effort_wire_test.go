package reasoning

import "testing"

func TestAcceptsEffortWire(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model  string
		accept bool
	}{
		{"qwen3-next-80b-a3b-thinking", false},
		{"dashscope/qwen3-next-80b-a3b-thinking", false},
		{"qwen3.7-max", false},
		{"qwen3.8-max", true},
		{"qwen3.8-flash", true},
		{"qwen3.8-2.4t-a95b", true},
		{"dashscope/qwen3.8-max", true},
		{"qwen3.5-plus", false},
		{"qwq-32b", false},
		{"kimi-k2.7-code-highspeed", false},
		{"moonshot/kimi-k2.7-code", false},
		{"dashscope/kimi-k2.7-code", false},
		{"kimi-k2.6", false},
		{"moonshot/kimi-k2.6", false},
		{"kimi-k2.5", false},
		{"kimi-k2-thinking", false},
		{"kimi-k3", true},
		{"moonshot/kimi-k3", true},
		{"gpt-5.5", true},
		{"gpt-4o", false},
		{"gpt-4o-mini", false},
		{"gpt-4.1", false},
		{"gpt-4-turbo", false},
		{"gpt-3.5-turbo", false},
		{"openai/gpt-4o", false},
		{"o3-mini", true},
		{"deepseek-v4-pro", true},
		{"glm-5-turbo", false},
		{"glm-5.1", false},
		{"zai/glm-5.1", false},
		{"glm-5", false},
		{"glm-4.7-flashx", false},
		{"glm-4.6v", false},
		{"glm-4.5-air", false},
		{"glm-5.2", true},
		{"zai/glm-5.2", true},
		{"glm-5.2-fast-preview", true},
		{"glm-5.3", true},
		{"glm-5.3-flash", true},
		{"glm-latest", true},
		{"glm-flash-latest", true},
		{"dashscope/glm-5.1", true},
		{"dashscope/glm-5", false},
		{"zai-glm-5-2", true},
		{"mistral/glm-5-2", true},
		{"minimax-m3", false},
		{"MiniMax-M3", false},
		{"minimax/MiniMax-M3", false},
		{"MiniMax-M2.7", false},
		{"MiniMax-M2.7-highspeed", false},
		{"minimax/MiniMax-M2.5", false},
	} {
		if got := AcceptsEffortWire(tc.model); got != tc.accept {
			t.Errorf("AcceptsEffortWire(%q) = %v, want %v", tc.model, got, tc.accept)
		}
	}
}

func TestTakesNoThinkingDepth(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model string
		want  bool
	}{
		{"glm-5.1", true},
		{"zai/glm-5.1", true},
		{"glm-5-turbo", true},
		{"glm-4.7", true},
		{"kimi-k2.6", true},
		{"moonshot/kimi-k2.6", true},
		{"kimi-k2.7-code", true},
		{"MiniMax-M3", true},
		{"minimax/MiniMax-M2.7", true},
		{"glm-5.2", false},
		{"zai/glm-5.2", false},
		{"glm-latest", false},
		{"kimi-k3", false},
		{"dashscope/glm-5.1", false},
		{"dashscope/glm-5", false},
		{"dashscope/kimi-k2.7-code", false},
		{"dashscope/kimi-k2.6", false},
		{"zai-glm-5-2", false},
		{"qwen3.7-plus", false},
		{"gpt-5", false},
		{"deepseek-v4-pro", false},
	} {
		if got := TakesNoThinkingDepth(tc.model); got != tc.want {
			t.Errorf("TakesNoThinkingDepth(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestRejectsMinP(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model  string
		reject bool
	}{
		{"claude-sonnet-4-5", true},
		{"claude-sonnet-4-5-20250929", true},
		{"anthropic/claude-sonnet-4.5", true},
		{"us.anthropic.claude-opus-4-6-v1:0", true},
		{"claude-opus-4-7", true},
		{"claude-haiku-4-5", true},
		{"gpt-5.5", true},
		{"gpt-4o", true},
		{"o3-mini", true},
		{"chatgpt-4o-latest", true},
		{"gpt-oss-120b", false},
		{"grok-4.6", false},
		{"qwen3-32b", false},
		{"deepseek-v3.2", false},
		{"zai/glm-5.3", false},
	} {
		if got := RejectsMinP(tc.model); got != tc.reject {
			t.Errorf("RejectsMinP(%q) = %v, want %v", tc.model, got, tc.reject)
		}
	}
}

func TestServedByDeepSeek(t *testing.T) {
	t.Parallel()

	const deepSeek, dashScope, gateway = "api.deepseek.com", "dashscope-us.aliyuncs.com", "litellm.example"
	for _, tc := range []struct {
		model, host string
		served      bool
	}{
		{"deepseek-flash", deepSeek, true},
		{"deepseek-v4-pro", deepSeek, true},
		{"DeepSeek-V4-Pro", deepSeek, true},
		{"deepseek-v4-flash", deepSeek, true},
		{"deepseek-v4-flash-vision-exp", deepSeek, true},
		{"deepseek/deepseek-v4-pro", gateway, true},
		{"deepseek/deepseek-flash", gateway, true},
		{"deepseek-v4-pro", dashScope, false},
		{"deepseek-v4-flash", dashScope, false},
		{"deepseek-v4-pro", gateway, false},
		{"deepseek-v4-pro", "", false},
		{"dashscope/deepseek-v4-pro", gateway, false},
		{"openrouter/deepseek/deepseek-v4-pro", gateway, false},
		{"deepseek-v4-flash-0731", deepSeek, false},
		{"deepseek-v4.1-flash", deepSeek, false},
		{"deepseek-r1", deepSeek, false},
		{"deepseek-v3.2", deepSeek, false},
		{"gpt-5.5", deepSeek, false},
	} {
		if got := ServedByDeepSeek(tc.model, tc.host); got != tc.served {
			t.Errorf("ServedByDeepSeek(%q, %q) = %v, want %v", tc.model, tc.host, got, tc.served)
		}
	}
}

func TestResolveOffOnDoorsThatRejectEffort(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model string
		want  OffWire
	}{
		{"qwen3-next-80b-a3b-thinking", OffUnsupported},
		{"kimi-k2.7-code-highspeed", OffUnsupported},
		{"kimi-k2.6", OffDisableThinkingObject},
		{"kimi-k2-thinking", OffUnsupported},
		{"kimi-k3", OffUnsupported},
		{"deepseek-v4-pro", OffDisableThinkingObject},
		{"glm-5-turbo", OffDisableThinkingObject},
	} {
		if got := ResolveOff(tc.model, ProviderOpenAI); got != tc.want {
			t.Errorf("ResolveOff(%q, ProviderOpenAI) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestTheGrokBuildAliasThatIsReallyGrok45(t *testing.T) {
	t.Parallel()

	if !AcceptsEffortWire("grok-build-latest") {
		t.Error("grok-build-latest drops the effort field, but the vendor answers low with 47 " +
			"reasoning tokens and high with 73")
	}
	if got := ResolveOff("grok-build-latest", ProviderOpenAI); got != OffUnsupported {
		t.Errorf("ResolveOff(grok-build-latest) = %v, want unsupported: the vendor refuses "+
			"effort none on this model", got)
	}

	for _, model := range []string{"grok-build-0.1", "grok-code-fast-1"} {
		if AcceptsEffortWire(model) {
			t.Errorf("%s carries the effort field, but the vendor refuses it by name", model)
		}
	}
}

func TestEffortWithTools(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model string
		rule  EffortToolsRule
	}{
		{"gpt-5.6-sol", EffortToolsDisable},
		{"gpt-5.6-terra", EffortToolsDisable},
		{"openai/gpt-5.6", EffortToolsDisable},
		{"gpt-5.5", EffortToolsOmit},
		{"gpt-5.4-nano", EffortToolsOmit},
		{"gpt-5.4-mini", EffortToolsOmit},
		{"gpt-5.2", EffortToolsFree},
		{"gpt-5.1", EffortToolsFree},
		{"gpt-5-mini", EffortToolsFree},
		{"gpt-4o", EffortToolsFree},
		{"claude-opus-5", EffortToolsFree},
	} {
		if got := EffortWithTools(tc.model); got != tc.rule {
			t.Errorf("EffortWithTools(%q) = %v, want %v", tc.model, got, tc.rule)
		}
	}
}
