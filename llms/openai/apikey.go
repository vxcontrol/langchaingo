package openai

import (
	"net/url"
	"strings"

	"github.com/vxcontrol/langchaingo/llms/openai/internal/openaiclient"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

// APIKeyRequiredBaseURLs lists documented OpenAI-compatible API base URLs of
// public providers that require an API key. Local and self-hosted backends
// (vLLM, Ollama, llama.cpp, SGLang, LiteLLM on a private host, etc.) are not
// listed and may be used without a key.
//
// Matching uses the hostname of these URLs (and Azure host suffixes), so
// common variants such as a trailing slash or an extra /v1 still require a key.
// An empty base URL is treated as the default OpenAI endpoint.
var APIKeyRequiredBaseURLs = reasoning.PublicProviderBaseURLs()

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
	return reasoning.PublicProviderHost(hostnameFromURL(baseURL))
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

// ServedByOpenAI reports whether a door built on baseURL calls OpenAI's own API: the default URL,
// api.openai.com, or a LiteLLM route that forwards to it unchanged.
func ServedByOpenAI(baseURL string) bool {
	host := hostnameFromURL(baseURL)
	return host == "" || reasoning.OpenAIHost(host) || openAIPassthrough(baseURL)
}

// openAIPassthrough reports a LiteLLM route that forwards to OpenAI's own API unchanged.
func openAIPassthrough(raw string) bool {
	raw = strings.TrimSpace(raw)
	if !strings.Contains(raw, "://") {
		raw = "https://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil || reasoning.PublicProviderHost(strings.ToLower(u.Hostname())) {
		return false
	}
	first, _, _ := strings.Cut(strings.Trim(u.Path, "/"), "/")
	return first == "openai_passthrough"
}
