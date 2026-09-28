package openai_test

import (
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func scenarios() []scen {
	or := "https://openrouter.ai/api/v1"
	mod := []openai.Option{openai.WithModernReasoningFormat()}
	ad := []llms.CallOption{llms.WithAdaptiveReasoning("")}
	return []scen{
		{name: "litellm-haiku45", base: "http://litellm.local:4000", model: "claude-haiku-4-5", opts: ad},
		{name: "litellm-sonnet45", base: "http://litellm.local:4000", model: "claude-sonnet-4-5", opts: ad},
		{name: "litellm-sonnet37", base: "http://litellm.local:4000", model: "claude-3-7-sonnet-latest", opts: ad},
		{name: "or-legacy-sonnet45", base: or, model: "anthropic/claude-sonnet-4.5", opts: ad},
		{name: "or-modern-sonnet45", base: or, model: "anthropic/claude-sonnet-4.5", cliOpts: mod, opts: ad},
		{name: "or-modern-opus48", base: or, model: "anthropic/claude-opus-4-8", cliOpts: mod, opts: ad},
		{name: "anthropic-compat-opus46", base: "https://api.anthropic.com/v1/", model: "claude-opus-4-6", opts: ad},
		{name: "or-modern-gemini25flash", base: or, model: "google/gemini-2.5-flash", cliOpts: mod, opts: ad},
	}
}
