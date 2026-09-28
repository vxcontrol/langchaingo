package openai_test

import (
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func scenarios() []scen {
	or := "https://openrouter.ai/api/v1"
	mod := []openai.Option{openai.WithModernReasoningFormat()}
	return []scen{
		{name: "or-modern-opus46-adaptive", base: or, model: "anthropic/claude-opus-4.6", cliOpts: mod, opts: []llms.CallOption{llms.WithAdaptiveReasoning("")}},
		{name: "or-modern-sonnet46-adaptive", base: or, model: "anthropic/claude-sonnet-4-6", cliOpts: mod, opts: []llms.CallOption{llms.WithAdaptiveReasoning("")}},
		{name: "or-legacy-opus46-adaptive", base: or, model: "anthropic/claude-opus-4.6", opts: []llms.CallOption{llms.WithAdaptiveReasoning("")}},
		{name: "litellm-opus46-adaptive", base: "http://litellm.local:4000", model: "claude-opus-4-6", opts: []llms.CallOption{llms.WithAdaptiveReasoning("")}},
		{name: "or-modern-gpt51-adaptive", base: or, model: "openai/gpt-5.1", cliOpts: mod, opts: []llms.CallOption{llms.WithAdaptiveReasoning("")}},
	}
}
