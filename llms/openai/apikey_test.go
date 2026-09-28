package openai

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRequiresAPIKey_ListedProviders(t *testing.T) {
	t.Parallel()

	for _, raw := range APIKeyRequiredBaseURLs {
		require.Truef(t, RequiresAPIKey(raw, APITypeOpenAI), "listed URL must require a key: %s", raw)
		require.Truef(t, RequiresAPIKey(raw+"/", APITypeOpenAI), "trailing slash must still require a key: %s", raw)
	}
}

func TestRequiresAPIKey_VariantsAndLocalBackends(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		baseURL  string
		apiType  APIType
		required bool
	}{
		{name: "empty defaults to openai", baseURL: "", required: true},
		{name: "whitespace defaults to openai", baseURL: "   ", required: true},
		{name: "openai without version path", baseURL: "https://api.openai.com", required: true},
		{name: "openai with extra path", baseURL: "https://api.openai.com/v1/chat", required: true},
		{name: "deepseek with v1 alias", baseURL: "https://api.deepseek.com/v1", required: true},
		{name: "novita v3 openai path", baseURL: "https://api.novita.ai/v3/openai", required: true},
		{name: "openrouter host only", baseURL: "https://openrouter.ai/api/v1/", required: true},
		{name: "azure resource host", baseURL: "https://my-resource.openai.azure.com", required: true},
		{name: "azure cognitive services", baseURL: "https://my-resource.cognitiveservices.azure.com", required: true},
		{name: "azure ai foundry", baseURL: "https://my-resource.services.ai.azure.com/models", required: true},
		{name: "azure api type ignores url", baseURL: "http://127.0.0.1:8000/v1", apiType: APITypeAzure, required: true},
		{name: "azure ad api type ignores url", baseURL: "http://127.0.0.1:8000/v1", apiType: APITypeAzureAD, required: true},
		{name: "vllm localhost", baseURL: "http://127.0.0.1:8000/v1", required: false},
		{name: "vllm custom host", baseURL: "http://vllm:8000/v1", required: false},
		{name: "ollama local", baseURL: "http://localhost:11434/v1", required: false},
		{name: "llamacpp", baseURL: "http://10.0.0.5:8080/v1", required: false},
		{name: "litellm private proxy", baseURL: "http://litellm-proxy:4000", required: false},
		{name: "scheme-less local host", baseURL: "192.168.1.10:8000/v1", required: false},
		{name: "lookalike openai host", baseURL: "https://api.openai.com.attacker.example/v1", required: false},
		{name: "baseten model apis", baseURL: "https://inference.baseten.co/v1", required: true},
		{name: "venice ai", baseURL: "https://api.venice.ai/api/v1", required: true},
		{name: "zenmux", baseURL: "https://zenmux.ai/api/v1", required: true},
		// Shared Cloudflare host must not require a key: Workers AI embeds
		// account IDs in the path and the same host serves unrelated APIs.
		{name: "cloudflare shared host", baseURL: "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1", required: false},
		// Scaleway dedicated inference uses a per-deployment UUID host.
		{name: "scaleway dedicated uuid host", baseURL: "https://01234567-89ab-cdef-0123-456789abcdef.ifr.fr-par.scaleway.com/v1", required: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tt.required, RequiresAPIKey(tt.baseURL, tt.apiType))
		})
	}
}

func TestNewClient_AllowsEmptyTokenForLocalBackends(t *testing.T) {
	t.Setenv(tokenEnvVarName, "")
	t.Setenv(baseURLEnvVarName, "")
	t.Setenv(baseAPIBaseEnvVarName, "")

	opts, cli, err := newClient(WithToken(""), WithBaseURL("http://127.0.0.1:8000/v1"))
	require.NoError(t, err)
	require.NotNil(t, opts)
	require.NotNil(t, cli)
	require.Empty(t, opts.token)
}

func TestNewClient_RequiresTokenForKnownProviders(t *testing.T) {
	t.Setenv(tokenEnvVarName, "")
	t.Setenv(baseURLEnvVarName, "")
	t.Setenv(baseAPIBaseEnvVarName, "")

	t.Run("default openai endpoint", func(t *testing.T) {
		_, _, err := newClient(WithToken(""))
		require.ErrorIs(t, err, ErrMissingToken)
	})

	t.Run("explicit openai url", func(t *testing.T) {
		_, _, err := newClient(WithToken(""), WithBaseURL("https://api.openai.com/v1"))
		require.ErrorIs(t, err, ErrMissingToken)
	})

	t.Run("openrouter", func(t *testing.T) {
		_, _, err := newClient(WithToken(""), WithBaseURL("https://openrouter.ai/api/v1"))
		require.ErrorIs(t, err, ErrMissingToken)
	})

	t.Run("azure api type", func(t *testing.T) {
		_, _, err := newClient(
			WithToken(""),
			WithAPIType(APITypeAzure),
			WithModel("gpt-4"),
			WithEmbeddingModel("text-embedding-3-small"),
			WithBaseURL("http://127.0.0.1:8000/v1"),
		)
		require.ErrorIs(t, err, ErrMissingToken)
	})
}

func TestHostnameFromURL(t *testing.T) {
	t.Parallel()

	require.Equal(t, "api.openai.com", hostnameFromURL("https://api.openai.com/v1/"))
	require.Equal(t, "api.openai.com", hostnameFromURL("API.OPENAI.COM/v1"))
	require.Equal(t, "127.0.0.1", hostnameFromURL("http://127.0.0.1:8000/v1"))
	require.Empty(t, hostnameFromURL(""))
}
