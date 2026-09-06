package openai

import (
	"net/url"
	"strings"

	"github.com/vxcontrol/langchaingo/llms/openai/internal/openaiclient"
)

// APIKeyRequiredBaseURLs lists documented OpenAI-compatible API base URLs of
// public providers that require an API key. Local and self-hosted backends
// (vLLM, Ollama, llama.cpp, SGLang, LiteLLM on a private host, etc.) are not
// listed and may be used without a key.
//
// Matching uses the hostname of these URLs (and Azure host suffixes), so
// common variants such as a trailing slash or an extra /v1 still require a key.
// An empty base URL is treated as the default OpenAI endpoint.
var APIKeyRequiredBaseURLs = []string{
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

	// PentAGI custom / aggregator backends (from project documentation).
	"https://openrouter.ai/api/v1",
	"https://api.deepinfra.com/v1/openai",
	"https://opencode.ai/zen/go/v1",
	"https://api.novita.ai/openai",
	"https://api.atlascloud.ai/v1",
	"https://api.orcarouter.ai/v1",
	"https://api.x.ai/v1",
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

// apiKeyRequiredHostSuffixes matches tenant-specific and regional hosts that
// cannot be listed as a single static URL (Azure OpenAI / Azure AI Foundry)
// and future OpenAI data-residency regions.
var apiKeyRequiredHostSuffixes = []string{
	".api.openai.com",
	".openai.azure.com",
	".cognitiveservices.azure.com",
	".services.ai.azure.com",
	".openai.azure.us",
	".openai.azure.cn",
}

var apiKeyRequiredHosts map[string]struct{}

func init() {
	apiKeyRequiredHosts = make(map[string]struct{}, len(APIKeyRequiredBaseURLs))
	for _, raw := range APIKeyRequiredBaseURLs {
		if host := hostnameFromURL(raw); host != "" {
			apiKeyRequiredHosts[host] = struct{}{}
		}
	}
}

// RequiresAPIKey reports whether the given base URL (and API type) belongs to a
// public OpenAI-compatible provider that requires an API key. Local and
// self-hosted backends such as vLLM do not.
func RequiresAPIKey(baseURL string, apiType APIType) bool {
	if openaiclient.IsAzure(openaiclient.APIType(apiType)) {
		return true
	}
	if strings.TrimSpace(baseURL) == "" {
		return true
	}
	host := hostnameFromURL(baseURL)
	if host == "" {
		return false
	}
	if _, ok := apiKeyRequiredHosts[host]; ok {
		return true
	}
	for _, suffix := range apiKeyRequiredHostSuffixes {
		if host == strings.TrimPrefix(suffix, ".") || strings.HasSuffix(host, suffix) {
			return true
		}
	}
	return false
}

func hostnameFromURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if !strings.Contains(raw, "://") {
		raw = "https://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return strings.ToLower(u.Hostname())
}
