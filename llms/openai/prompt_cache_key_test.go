package openai

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestThePromptCacheKeyReachesTheHostsThatDocumentIt(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		baseURL, model string
		sent           bool
	}{
		"openai":                 {"http://api.openai.com/v1", "gpt-5.5", true},
		"openai region":          {"http://eu.api.openai.com/v1", "gpt-5.6-sol", true},
		"xai":                    {"http://api.x.ai/v1", "grok-4.7", true},
		"mistral":                {"http://api.mistral.ai/v1", "mistral-large-latest", true},
		"moonshot":               {"http://api.moonshot.ai/v1", "kimi-k3", true},
		"gateway xai route":      {gatewayBaseURL, "xai/grok-4.7", true},
		"gateway mistral route":  {gatewayBaseURL, "mistral/mistral-large-latest", true},
		"gateway moonshot route": {gatewayBaseURL, "moonshot/kimi-k3", true},
		"gateway bare name":      {gatewayBaseURL, "gpt-5.5", false},
		"gateway openai route":   {gatewayBaseURL, "openai/gpt-5.5", false},
		"deepseek":               {deepSeekBaseURL, "deepseek-chat", false},
		"dashscope":              {dashScopeBaseURL, "qwen-plus", false},
		"dashscope kimi":         {dashScopeBaseURL, "kimi-k2.6", false},
		"z.ai":                   {"http://api.z.ai/api/paas/v4", "glm-5.3", false},
		"minimax":                {miniMaxHostURL, "MiniMax-M3", false},
		"openrouter":             {openRouterBaseURL, "openai/gpt-5.5", false},
		"openrouter grok":        {openRouterBaseURL, "x-ai/grok-4.7", false},
		"azure":                  {"http://pentagi.openai.azure.com/openai/v1", "gpt-5.5", false},
		"vllm":                   {"http://localhost:8000/v1", "gpt-oss-120b", false},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			body, _ := sendToHost(t, tc.baseURL, tc.model, llms.WithPromptCacheKey("flow-7:chain-42"))
			if tc.sent {
				require.Equal(t, "flow-7:chain-42", body["prompt_cache_key"])
			} else {
				require.NotContains(t, body, "prompt_cache_key")
			}

			body, _ = sendToHost(t, tc.baseURL, tc.model)
			require.NotContains(t, body, "prompt_cache_key")
		})
	}
}

func TestNoCacheLayoutTurnsOffTheWriteOpenAIMakesOnItsOwn(t *testing.T) {
	t.Parallel()

	const openAI = "http://api.openai.com/v1"
	explicit := map[string]any{"mode": "explicit"}
	none := llms.WithCacheLayout(llms.CacheLayoutNone)
	for name, tc := range map[string]struct {
		baseURL, model string
		opts           []llms.CallOption
		want           any
	}{
		"gpt-5.6":           {openAI, "gpt-5.6-terra", []llms.CallOption{none}, explicit},
		"gpt-6.1":           {openAI, "gpt-6.1-sol", []llms.CallOption{none}, explicit},
		"a later pro":       {openAI, "gpt-6.2-pro", []llms.CallOption{none}, explicit},
		"gpt-5.5":           {openAI, "gpt-5.5", []llms.CallOption{none}, nil},
		"gpt-5.4-mini":      {openAI, "gpt-5.4-mini", []llms.CallOption{none}, nil},
		"the door's layout": {openAI, "gpt-5.6-terra", nil, nil},
		"a growing history": {openAI, "gpt-5.6-terra", []llms.CallOption{llms.WithCacheLayout(llms.CacheLayoutGrowing)}, nil},
		"a gateway":         {gatewayBaseURL, "gpt-5.6-terra", []llms.CallOption{none}, nil},
		"azure":             {"http://pentagi.openai.azure.com/openai/v1", "gpt-5.6-terra", []llms.CallOption{none}, nil},
		"openrouter":        {openRouterBaseURL, "openai/gpt-5.6-terra", []llms.CallOption{none}, nil},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			body, _ := sendToHost(t, tc.baseURL, tc.model, tc.opts...)
			require.Equal(t, tc.want, body["prompt_cache_options"])
		})
	}
}
