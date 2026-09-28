package openai_test

import (
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func scenarios() []scen {
	or := "https://openrouter.ai/api/v1"
	return []scen{
		{name: "gpt51-effort-none", model: "gpt-5.1", opts: []llms.CallOption{llms.WithReasoning("none", 0)}},
		{name: "gpt52-effort-none-temp", model: "gpt-5.2", opts: []llms.CallOption{llms.WithReasoning("none", 0), llms.WithTemperature(0.3)}},
		{name: "or-modern-gpt51-none", base: or, model: "openai/gpt-5.1", cliOpts: []openai.Option{openai.WithModernReasoningFormat()}, opts: []llms.CallOption{llms.WithReasoning("none", 0)}},
		{name: "gpt5-minimal", model: "gpt-5", opts: []llms.CallOption{llms.WithReasoning("minimal", 0)}},
	}
}
