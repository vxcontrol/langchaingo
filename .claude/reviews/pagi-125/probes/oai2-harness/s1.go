package openai_test

import (
	"github.com/vxcontrol/langchaingo/llms"
)

func scenarios() []scen {
	return []scen{
		{name: "gpt5-temp", model: "gpt-5", opts: []llms.CallOption{llms.WithTemperature(0.2)}},
		{name: "gpt51-temp", model: "gpt-5.1", opts: []llms.CallOption{llms.WithTemperature(0.2)}},
		{name: "o3-temp", model: "o3", opts: []llms.CallOption{llms.WithTemperature(0.2)}},
		{name: "gpt4o-topk", model: "gpt-4o", opts: []llms.CallOption{llms.WithTemperature(0.2), llms.WithTopK(5)}},
		{name: "ds-chat-or", base: "https://openrouter.ai/api/v1", model: "deepseek/deepseek-chat", opts: []llms.CallOption{llms.WithTemperature(0.2), llms.WithTopP(0.9), llms.WithMaxTokens(100)}},
		{name: "ds-flash-or", base: "https://openrouter.ai/api/v1", model: "deepseek/deepseek-v4-flash", opts: []llms.CallOption{llms.WithTemperature(0.2), llms.WithTopP(0.9), llms.WithTopK(3), llms.WithMaxTokens(100)}},
		{name: "grok-temp", model: "grok-4", opts: []llms.CallOption{llms.WithTemperature(0.2), llms.WithMaxTokens(100)}},
	}
}
