package reasoning

import (
	"net/url"
	"slices"
	"strings"
)

type Vendor int

const (
	// VendorUnknown is a host that serves the name under its own catalogue:
	// vLLM, LM Studio, a public provider.
	VendorUnknown Vendor = iota
	VendorDeepSeek
	VendorZAI
	VendorMoonshot
	VendorXAI
	VendorMistral
	VendorDashScope
	VendorMiniMax
)

var vendorHosts = map[string]Vendor{
	"api.deepseek.com":                   VendorDeepSeek,
	"api.z.ai":                           VendorZAI,
	"open.bigmodel.cn":                   VendorZAI,
	"api.moonshot.ai":                    VendorMoonshot,
	"api.moonshot.cn":                    VendorMoonshot,
	"api.x.ai":                           VendorXAI,
	"us.api.x.ai":                        VendorXAI,
	"api.mistral.ai":                     VendorMistral,
	"codestral.mistral.ai":               VendorMistral,
	"dashscope.aliyuncs.com":             VendorDashScope,
	"dashscope-intl.aliyuncs.com":        VendorDashScope,
	"dashscope-us.aliyuncs.com":          VendorDashScope,
	"cn-hongkong.dashscope.aliyuncs.com": VendorDashScope,
	"api.minimax.io":                     VendorMiniMax,
	"api.minimax.cn":                     VendorMiniMax,
	"api.minimaxi.com":                   VendorMiniMax,
}

var litellmRoutes = map[string]Vendor{
	"deepseek":  VendorDeepSeek,
	"zai":       VendorZAI,
	"moonshot":  VendorMoonshot,
	"xai":       VendorXAI,
	"mistral":   VendorMistral,
	"dashscope": VendorDashScope,
	"minimax":   VendorMiniMax,
}

// ServedBy reports the vendor whose own API serves model on host: the owner of
// the host, or, off public providers, the vendor a LiteLLM route prefix names.
func ServedBy(model, host string) Vendor {
	if vendor := vendorOfHost(host); vendor != VendorUnknown {
		return vendor
	}
	if PublicProviderHost(host) {
		return VendorUnknown
	}
	route, _, routed := strings.Cut(strings.ToLower(model), "/")
	if !routed {
		return VendorUnknown
	}
	return litellmRoutes[route]
}

func vendorOfHost(host string) Vendor {
	if strings.HasSuffix(host, ".maas.aliyuncs.com") {
		return VendorDashScope
	}
	return vendorHosts[host]
}

// publicProviderBaseURLs lists documented OpenAI-compatible API base URLs of
// public providers that require an API key. Local and self-hosted backends
// (vLLM, Ollama, llama.cpp, SGLang, LiteLLM on a private host, etc.) are not
// listed and may be used without a key. ServedBy reads the ids on a listed host
// as its own catalogue, not as LiteLLM routes.
var publicProviderBaseURLs = []string{
	// OpenAI (default and data-residency regional endpoints).
	"https://api.openai.com/v1",
	"https://us.api.openai.com/v1",
	"https://eu.api.openai.com/v1",
	"https://au.api.openai.com/v1",
	"https://ca.api.openai.com/v1",
	"https://jp.api.openai.com/v1",
	"https://in.api.openai.com/v1",
	"https://sg.api.openai.com/v1",
	"https://kr.api.openai.com/v1",
	"https://gb.api.openai.com/v1",
	"https://ae.api.openai.com/v1",

	// Vendor APIs and aggregators PentAGI works with.
	"https://openrouter.ai/api/v1",
	"https://api.deepinfra.com/v1/openai",
	"https://opencode.ai/zen/go/v1",
	"https://api.novita.ai/openai",
	"https://api.atlascloud.ai/v1",
	"https://api.orcarouter.ai/v1",
	"https://api.x.ai/v1",
	"https://us.api.x.ai/v1",
	"https://api.deepseek.com",
	"https://api.z.ai/api/paas/v4",
	"https://api.z.ai/api/coding/paas/v4",
	"https://open.bigmodel.cn/api/paas/v4",
	"https://api.moonshot.ai/v1",
	"https://api.moonshot.cn/v1",
	"https://dashscope-us.aliyuncs.com/compatible-mode/v1",
	"https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
	"https://dashscope.aliyuncs.com/compatible-mode/v1",
	"https://api.minimax.io/v1",
	"https://api.minimax.cn/v1",
	"https://integrate.api.nvidia.com/v1",
	"https://api.hcnsec.cn/v1",
	"https://ollama.com/v1",

	// Additional public OpenAI-compatible providers.
	"https://api.groq.com/openai/v1",
	"https://api.together.ai/v1",
	"https://api.together.xyz/v1",
	"https://api.fireworks.ai/inference/v1",
	"https://api.cerebras.ai/v1",
	"https://api.sambanova.ai/v1",
	"https://api.mistral.ai/v1",
	"https://codestral.mistral.ai/v1",
	"https://api.perplexity.ai",
	"https://api.studio.nebius.ai/v1",
	"https://api.tokenfactory.nebius.com/v1",
	"https://router.huggingface.co/v1",
	"https://models.github.ai/inference",
	"https://models.inference.ai.azure.com",
	"https://generativelanguage.googleapis.com/v1beta/openai",
	"https://ai-gateway.vercel.sh/v1",
	"https://api.siliconflow.cn/v1",
	"https://api.siliconflow.com/v1",
	"https://api-inference.modelscope.cn/v1",
	"https://ark.cn-beijing.volces.com/api/v3",
	"https://api.minimaxi.com/v1",
	"https://api.cohere.ai/compatibility/v1",
	"https://api.cohere.com/compatibility/v1",
	"https://api.hyperbolic.xyz/v1",
	"https://api.lambda.ai/v1",
	"https://api.lambdalabs.com/v1",
	"https://api.friendli.ai/serverless/v1",
	"https://api.scaleway.ai/v1",
	"https://inference.baseten.co/v1",
	"https://api.venice.ai/api/v1",
	"https://zenmux.ai/api/v1",
}

// publicProviderHostSuffixes matches tenant-specific and regional hosts that
// cannot be listed as a single static URL (Azure OpenAI / Azure AI Foundry)
// and future OpenAI data-residency regions.
var publicProviderHostSuffixes = []string{
	".api.openai.com",
	".openai.azure.com",
	".cognitiveservices.azure.com",
	".services.ai.azure.com",
	".openai.azure.us",
	".openai.azure.cn",
}

var publicProviderHosts = func() map[string]bool {
	hosts := make(map[string]bool, len(publicProviderBaseURLs))
	for _, raw := range publicProviderBaseURLs {
		if u, err := url.Parse(raw); err == nil {
			hosts[strings.ToLower(u.Hostname())] = true
		}
	}
	return hosts
}()

// PublicProviderBaseURLs returns a copy of the base URLs PublicProviderHost matches.
func PublicProviderBaseURLs() []string {
	return slices.Clone(publicProviderBaseURLs)
}

// PublicProviderHost reports whether a lowercased hostname belongs to a public
// OpenAI-compatible provider.
func PublicProviderHost(host string) bool {
	if host == "" {
		return false
	}
	if publicProviderHosts[host] || vendorOfHost(host) != VendorUnknown {
		return true
	}
	for _, suffix := range publicProviderHostSuffixes {
		if host == strings.TrimPrefix(suffix, ".") || strings.HasSuffix(host, suffix) {
			return true
		}
	}
	return false
}
