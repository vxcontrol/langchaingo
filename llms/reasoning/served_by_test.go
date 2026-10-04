package reasoning

import "testing"

func TestServedBy(t *testing.T) {
	t.Parallel()

	const gateway, vllm = "litellm.internal", "vllm.internal"
	for _, tc := range []struct {
		model, host string
		want        Vendor
	}{
		{"deepseek-v4-pro", "api.deepseek.com", VendorDeepSeek},
		{"glm-5.3", "api.z.ai", VendorZAI},
		{"glm-4.6", "open.bigmodel.cn", VendorZAI},
		{"kimi-k2.6", "api.moonshot.cn", VendorMoonshot},
		{"grok-4.7", "api.x.ai", VendorXAI},
		{"grok-4.7", "us.api.x.ai", VendorXAI},
		{"mistral-large-latest", "api.mistral.ai", VendorMistral},
		{"codestral-latest", "codestral.mistral.ai", VendorMistral},
		{"qwen3.6-plus", "dashscope-intl.aliyuncs.com", VendorDashScope},
		{"qwen3.6-plus", "ws-1.cn-beijing.maas.aliyuncs.com", VendorDashScope},
		{"zai/glm-5.3", "api.deepseek.com", VendorDeepSeek},

		{"deepseek/deepseek-v4-pro", gateway, VendorDeepSeek},
		{"ZAI/glm-5.3", gateway, VendorZAI},
		{"moonshot/kimi-k2.6", gateway, VendorMoonshot},
		{"xai/grok-4.7", gateway, VendorXAI},
		{"mistral/mistral-large-latest", gateway, VendorMistral},
		{"dashscope/qwen3.6-plus", gateway, VendorDashScope},
		{"deepseek/deepseek-v4-pro", "", VendorDeepSeek},

		{"deepseek-v4-pro", gateway, VendorUnknown},
		{"glm-4.6", vllm, VendorUnknown},
		{"openrouter/deepseek/deepseek-v4-pro", gateway, VendorUnknown},
		{"zai-org/GLM-4.6", vllm, VendorUnknown},
		{"deepseek/deepseek-v4-pro", "openrouter.ai", VendorUnknown},
		{"zai/glm-4.6", "ai-gateway.vercel.sh", VendorUnknown},
		{"deepseek/deepseek-v4-pro", "zenmux.ai", VendorUnknown},
		{"deepseek/deepseek-v4-pro", "api.novita.ai", VendorUnknown},
		{"xai/grok-4.7", "api.orcarouter.ai", VendorUnknown},
		{"gpt-5.5", "api.openai.com", VendorUnknown},
	} {
		if got := ServedBy(tc.model, tc.host); got != tc.want {
			t.Errorf("ServedBy(%q, %q) = %v, want %v", tc.model, tc.host, got, tc.want)
		}
	}
}

func TestPublicProviderHost(t *testing.T) {
	t.Parallel()

	for host, want := range map[string]bool{
		"openrouter.ai":                    true,
		"ai-gateway.vercel.sh":             true,
		"us.api.x.ai":                      true,
		"codestral.mistral.ai":             true,
		"eu.api.openai.com":                true,
		"tenant.openai.azure.com":          true,
		"llm.pentagi.net":                  false,
		"litellm.internal":                 false,
		"localhost":                        false,
		"":                                 false,
		"api.openai.com.attacker.internal": false,
	} {
		if got := PublicProviderHost(host); got != want {
			t.Errorf("PublicProviderHost(%q) = %v, want %v", host, got, want)
		}
	}
}

func TestDashScopeRoute(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ model, host, want string }{
		{"qwen3.6-plus", "dashscope-intl.aliyuncs.com", "dashscope/qwen3.6-plus"},
		{"dashscope/qwen3.6-plus", "dashscope-intl.aliyuncs.com", "dashscope/qwen3.6-plus"},
		{"dashscope/qwen3.6-plus", "litellm.internal", "dashscope/qwen3.6-plus"},
		{"qwen3.6-plus", "litellm.internal", "qwen3.6-plus"},
		{"qwen3.6-plus", "vllm.internal", "qwen3.6-plus"},
		{"dashscope/qwq-plus", "ai-gateway.vercel.sh", "qwq-plus"},
		{"DashScope/QwQ-Plus", "openrouter.ai", "QwQ-Plus"},
	} {
		if got := DashScopeRoute(tc.model, tc.host); got != tc.want {
			t.Errorf("DashScopeRoute(%q, %q) = %q, want %q", tc.model, tc.host, got, tc.want)
		}
	}
}
