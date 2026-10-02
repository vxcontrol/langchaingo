package openai

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestAForcedToolChoiceTheVendorRejectsIsRefusedBeforeTheNetwork(t *testing.T) {
	t.Parallel()

	named := llms.WithToolChoice(map[string]any{"type": "function", "name": "lookup"})
	required := llms.WithToolChoice("required")
	thinking := llms.WithReasoning(llms.ReasoningHigh, 0)
	call := func(baseURL, model string, opts ...llms.CallOption) (int, error) {
		doer := &bodyDoer{}
		llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "look it up")},
			append([]llms.CallOption{llms.WithTools([]llms.Tool{astraTool()})}, opts...)...)
		if doer.body == nil {
			return 0, err
		}
		return 1, err
	}

	const (
		deepSeek   = "https://api.deepseek.com"
		dashScope  = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
		moonshot   = "https://api.moonshot.ai/v1"
		zai        = "https://api.z.ai/api/paas/v4"
		gateway    = "http://litellm.internal/v1"
		openRouter = "https://openrouter.ai/api/v1"
		vllm       = "http://vllm.internal:8000/v1"
	)
	requiredInTheExtraBody := llms.WithExtraBody(map[string]any{"tool_choice": "required"})
	for name, tc := range map[string]struct {
		baseURL, model string
		opts           []llms.CallOption
	}{
		"DeepSeek thinking, required":                   {deepSeek, "deepseek-v4-pro", []llms.CallOption{thinking, required}},
		"DeepSeek thinking, required in the extra body": {deepSeek, "deepseek-v4-pro", []llms.CallOption{thinking, requiredInTheExtraBody}},
		"Qwen thinking, named":                          {dashScope, "qwen3.6-plus", []llms.CallOption{thinking, named}},
		"Qwen switched on by the door, required":        {dashScope, "qwen-plus", []llms.CallOption{thinking, required}},
		"Qwen switched on by the extra body, required": {dashScope, "qwen-plus",
			[]llms.CallOption{llms.WithExtraBody(map[string]any{"enable_thinking": true}), required}},
		"Qwen through the gateway's dashscope route": {gateway, "dashscope/qwen3.6-plus", []llms.CallOption{thinking, named}},
		"Kimi thinking, named":                       {moonshot, "kimi-k2.5", []llms.CallOption{thinking, named}},
		"Kimi K2.6, required":                        {moonshot, "kimi-k2.6", []llms.CallOption{required}},
		"Kimi K2.7 Code, required":                   {moonshot, "kimi-k2.7-code", []llms.CallOption{required}},
		"Kimi through the gateway's moonshot route":  {gateway, "moonshot/kimi-k2.6", []llms.CallOption{required}},
		"GLM on Z.ai, required":                      {zai, "glm-4.6", []llms.CallOption{required}},
		"GLM on Z.ai, named":                         {zai, "glm-4.6", []llms.CallOption{named}},
		"GLM through the gateway's zai route":        {gateway, "zai/glm-4.6", []llms.CallOption{required}},
	} {
		calls, err := call(tc.baseURL, tc.model, tc.opts...)
		var refused *reasoning.ErrForcedToolChoiceUnsupported
		require.True(t, errors.As(err, &refused), "%s: %v", name, err)
		require.Zero(t, calls, "%s: refused before the network", name)
	}

	for name, tc := range map[string]struct {
		baseURL, model string
		opts           []llms.CallOption
	}{
		"Qwen with thinking off, named": {dashScope, "qwen3.6-plus", []llms.CallOption{llms.WithReasoningDisabled(), named}},
		"Qwen thinking off by the extra body, required": {dashScope, "qwen3.6-plus",
			[]llms.CallOption{llms.WithExtraBody(map[string]any{"enable_thinking": false}), required}},
		"Qwen3 the door keeps from thinking off a stream, required": {dashScope, "qwen3-32b", []llms.CallOption{required}},
		"Kimi thinking, required":                                   {moonshot, "kimi-k2.5", []llms.CallOption{thinking, required}},
		"GLM on a gateway, required":                                {gateway, "glm-4.6", []llms.CallOption{required}},
		"DeepSeek thinking, auto":                                   {deepSeek, "deepseek-v4-pro", []llms.CallOption{thinking, llms.WithToolChoice("auto")}},
		"GPT, required":                                             {"https://api.openai.com/v1", "gpt-5.4", []llms.CallOption{required}},
		"DeepSeek weights on vLLM":                                  {vllm, "deepseek-v4-pro", []llms.CallOption{thinking, required}},
		"Qwen on OpenRouter":                                        {openRouter, "qwen/qwen3.6-plus", []llms.CallOption{thinking, required}},
		"Kimi K2.6 on OpenRouter":                                   {openRouter, "moonshotai/kimi-k2.6", []llms.CallOption{required}},
		"Qwen weights on vLLM, named":                               {vllm, "Qwen/Qwen3-32B", []llms.CallOption{thinking, named}},
	} {
		calls, err := call(tc.baseURL, tc.model, tc.opts...)
		require.NoError(t, err, name)
		require.Equal(t, 1, calls, name)
	}
}
