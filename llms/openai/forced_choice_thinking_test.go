package openai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func callWithATool(t *testing.T, baseURL, model string, opts ...llms.CallOption) (map[string]any, error) {
	t.Helper()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "look it up")},
		append([]llms.CallOption{llms.WithTools([]llms.Tool{astraTool()})}, opts...)...)
	if doer.body == nil {
		return nil, err
	}
	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	return body, err
}

func TestAForcedToolChoiceTheVendorRejectsIsRefusedBeforeTheNetwork(t *testing.T) {
	t.Parallel()

	named := llms.WithToolChoice(map[string]any{"type": "function", "name": "lookup"})
	required := llms.WithToolChoice("required")
	thinking := llms.WithReasoning(llms.ReasoningHigh, 0)

	const (
		deepSeek   = "https://api.deepseek.com"
		dashScope  = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
		moonshot   = "https://api.moonshot.ai/v1"
		zai        = "https://api.z.ai/api/paas/v4"
		gateway    = "http://litellm.internal/v1"
		openRouter = "https://openrouter.ai/api/v1"
		vllm       = "http://vllm.internal:8000/v1"
		vercel     = "https://ai-gateway.vercel.sh/v1"
		zenMux     = "https://zenmux.ai/api/v1"
		novita     = "https://api.novita.ai/openai"
	)
	requiredInTheExtraBody := llms.WithExtraBody(map[string]any{"tool_choice": "required"})
	for name, tc := range map[string]struct {
		baseURL, model string
		opts           []llms.CallOption
	}{
		"DeepSeek thinking, required":                   {deepSeek, "deepseek-v4-pro", []llms.CallOption{thinking, required}},
		"DeepSeek thinking, required in the extra body": {deepSeek, "deepseek-v4-pro", []llms.CallOption{thinking, requiredInTheExtraBody}},
		"Qwen thinking, named":                          {dashScope, "qwen3.6-plus", []llms.CallOption{thinking, named}},
		"QwQ thinking, required":                        {dashScope, "qwq-plus", []llms.CallOption{thinking, required}},
		"Qwen switched on by the door, required":        {dashScope, "qwen-plus", []llms.CallOption{thinking, required}},
		"Qwen switched on by the extra body, required": {dashScope, "qwen-plus",
			[]llms.CallOption{llms.WithExtraBody(map[string]any{"enable_thinking": true}), required}},
		"Qwen through the gateway's dashscope route": {gateway, "dashscope/qwen3.6-plus", []llms.CallOption{thinking, named}},
		"Kimi thinking, named":                       {moonshot, "kimi-k2.5", []llms.CallOption{thinking, named}},
		"Kimi K2.6, required":                        {moonshot, "kimi-k2.6", []llms.CallOption{required}},
		"Kimi K2.6 on the China host, required":      {"https://api.moonshot.cn/v1", "kimi-k2.6", []llms.CallOption{required}},
		"Kimi K2.7 Code, required":                   {moonshot, "kimi-k2.7-code", []llms.CallOption{required}},
		"Kimi through the gateway's moonshot route":  {gateway, "moonshot/kimi-k2.6", []llms.CallOption{required}},
		"GLM on Z.ai, required":                      {zai, "glm-4.6", []llms.CallOption{required}},
		"GLM on Z.ai, named":                         {zai, "glm-4.6", []llms.CallOption{named}},
		"GLM through the gateway's zai route":        {gateway, "zai/glm-4.6", []llms.CallOption{required}},
	} {
		body, err := callWithATool(t, tc.baseURL, tc.model, tc.opts...)
		var refused *reasoning.ErrForcedToolChoiceUnsupported
		require.True(t, errors.As(err, &refused), "%s: %v", name, err)
		require.Nil(t, body, "%s: refused before the network", name)
	}

	namedOnTheWire := map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}}
	for name, tc := range map[string]struct {
		baseURL, model string
		opts           []llms.CallOption
		sent           any
	}{
		"Qwen with thinking off, named": {dashScope, "qwen3.6-plus", []llms.CallOption{llms.WithReasoningDisabled(), named}, namedOnTheWire},
		"Qwen thinking off by the extra body, required": {dashScope, "qwen3.6-plus",
			[]llms.CallOption{llms.WithExtraBody(map[string]any{"enable_thinking": false}), required}, "required"},
		"Qwen3 the door keeps from thinking off a stream, required": {dashScope, "qwen3-32b", []llms.CallOption{required}, "required"},
		"Kimi thinking, required":                                   {moonshot, "kimi-k2.5", []llms.CallOption{thinking, required}, "required"},
		"GLM on a gateway, required":                                {gateway, "glm-4.6", []llms.CallOption{required}, "required"},
		"DeepSeek thinking, auto":                                   {deepSeek, "deepseek-v4-pro", []llms.CallOption{thinking, llms.WithToolChoice("auto")}, "auto"},
		"GPT, required":                                             {"https://api.openai.com/v1", "gpt-5.4", []llms.CallOption{required}, "required"},
		"DeepSeek weights on vLLM":                                  {vllm, "deepseek-v4-pro", []llms.CallOption{thinking, required}, "required"},
		"Qwen on OpenRouter":                                        {openRouter, "qwen/qwen3.6-plus", []llms.CallOption{thinking, required}, "required"},
		"Kimi K2.6 on OpenRouter":                                   {openRouter, "moonshotai/kimi-k2.6", []llms.CallOption{required}, "required"},
		"Qwen weights on vLLM, named":                               {vllm, "Qwen/Qwen3-32B", []llms.CallOption{thinking, named}, namedOnTheWire},
		"Qwen by bare name on a gateway":                            {gateway, "qwen3.6-plus", []llms.CallOption{thinking, required}, "required"},
		"Qwen switched off by the extra body over the door": {dashScope, "qwen-plus",
			[]llms.CallOption{thinking, llms.WithExtraBody(map[string]any{"enable_thinking": false}), required}, "required"},
		"GLM with auto in the extra body over required": {zai, "glm-4.6",
			[]llms.CallOption{required, llms.WithExtraBody(map[string]any{"tool_choice": "auto"})}, "auto"},
		"Kimi on OpenRouter, named": {openRouter, "moonshotai/kimi-k2.5", []llms.CallOption{thinking, named}, namedOnTheWire},
		"DeepSeek V4 on OpenRouter": {openRouter, "deepseek/deepseek-v4-pro", []llms.CallOption{thinking, required}, "required"},
		"Kimi newer than the tables, thinking off by the extra body, named": {moonshot, "kimi-k4",
			[]llms.CallOption{llms.WithExtraBody(map[string]any{"thinking": map[string]any{"type": "disabled"}}), named},
			namedOnTheWire},
		"GLM ids on Vercel AI Gateway":              {vercel, "zai/glm-4.6", []llms.CallOption{required}, "required"},
		"DeepSeek ids on Vercel AI Gateway":         {vercel, "deepseek/deepseek-v4-pro", []llms.CallOption{thinking, required}, "required"},
		"DeepSeek ids on ZenMux":                    {zenMux, "deepseek/deepseek-v4-flash", []llms.CallOption{thinking, required}, "required"},
		"DeepSeek ids on Novita, named":             {novita, "deepseek/deepseek-v4-pro", []llms.CallOption{thinking, named}, namedOnTheWire},
		"DeepSeek ids on Novita, required":          {novita, "deepseek/deepseek-v4-flash", []llms.CallOption{thinking, required}, "required"},
		"GLM through the gateway's zai route, auto": {gateway, "zai/glm-4.6", []llms.CallOption{llms.WithToolChoice("auto")}, "auto"},
	} {
		body, err := callWithATool(t, tc.baseURL, tc.model, tc.opts...)
		require.NoError(t, err, name)
		require.NotNil(t, body, name)
		assert.Equal(t, tc.sent, body["tool_choice"], name)
	}
}
