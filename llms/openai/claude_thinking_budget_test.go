package openai

import (
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAClaudeBudgetTravelsAsAnthropicsThinkingObject(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-haiku-4-5", "anthropic/claude-sonnet-4-5", "anthropic/claude-sonnet-4-6"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			body := captureWireWithClient(t, model, nil,
				llms.WithReasoning(llms.ReasoningNone, 2048), llms.WithMaxTokens(8192))

			if !strings.Contains(body, `"thinking":{"type":"enabled","budget_tokens":2048}`) {
				t.Errorf("want the budget in the thinking object, got body: %s", body)
			}
			if strings.Contains(body, "reasoning_effort") {
				t.Errorf("an effort would replace the caller's budget on this route, got body: %s", body)
			}
			if !strings.Contains(body, `"max_completion_tokens":8192`) {
				t.Errorf("an answer limit above the budget must reach the wire verbatim, got body: %s", body)
			}
		})
	}
}

func TestTheAnswerLimitRisesUnderTheBudgetTheThinkingObjectCarries(t *testing.T) {
	t.Parallel()

	body := captureWireWithClient(t, "claude-haiku-4-5", nil,
		llms.WithReasoning(llms.ReasoningNone, 1024), llms.WithMaxTokens(1024))

	if !strings.Contains(body, `"thinking":{"type":"enabled","budget_tokens":1024}`) {
		t.Fatalf("want the vendor-floor budget in the thinking object, got body: %s", body)
	}
	if !strings.Contains(body, `"max_completion_tokens":2048`) {
		t.Errorf("the budget must be less than the answer limit, got body: %s", body)
	}
}

func TestAnAdaptiveOnlyClaudeGetsNoBudgetObject(t *testing.T) {
	t.Parallel()

	body := captureWireWithClient(t, "anthropic/claude-sonnet-5", nil,
		llms.WithReasoning(llms.ReasoningNone, 2048), llms.WithMaxTokens(8192))

	if strings.Contains(body, `"thinking"`) {
		t.Errorf("this generation rejects thinking.type enabled, got body: %s", body)
	}
}

func TestAnAdaptivePreferenceKeepsTheEffortOnAGenerationThatTakesBoth(t *testing.T) {
	t.Parallel()

	adaptiveWithBudget := func(o *llms.CallOptions) {
		o.Reasoning = &llms.ReasoningConfig{Adaptive: true, Effort: llms.ReasoningHigh, Tokens: 2048}
	}
	body := captureWireWithClient(t, "anthropic/claude-opus-4-6", nil, adaptiveWithBudget, llms.WithMaxTokens(8192))

	if strings.Contains(body, `"thinking"`) {
		t.Errorf("the caller asked for adaptive thinking, got body: %s", body)
	}
	if !strings.Contains(body, `"reasoning_effort":"high"`) {
		t.Errorf("want the effort that carries adaptive thinking, got body: %s", body)
	}
}

func TestABudgetStaysOffModelsThatAreNotClaude(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-5", "deepseek-v4-pro", "glm-5.2"} {
		body := captureWireWithClient(t, model, nil,
			llms.WithReasoning(llms.ReasoningNone, 2048), llms.WithMaxTokens(8192))
		if strings.Contains(body, `"budget_tokens"`) {
			t.Errorf("%s: Anthropic's budget field is Claude's alone, got body: %s", model, body)
		}
	}
}

func TestAClaudeBudgetStaysOffHostsThatDocumentNoThinkingObject(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct{ baseURL, model string }{
		"the gateway's deepinfra route":  {gatewayBaseURL, "deepinfra/anthropic/claude-sonnet-4-6"},
		"the gateway's perplexity route": {gatewayBaseURL, "perplexity/anthropic/claude-sonnet-4-5"},
		"the gateway's openrouter route": {gatewayBaseURL, "openrouter/anthropic/claude-haiku-4.5"},
		"DeepInfra":                      {"http://api.deepinfra.com/v1/openai", "anthropic/claude-sonnet-4-6"},
		"Perplexity":                     {"http://api.perplexity.ai", "anthropic/claude-sonnet-4-5"},
		"OpenRouter":                     {"http://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.5"},
		"ZenMux":                         {"http://zenmux.ai/api/v1", "anthropic/claude-sonnet-4.6"},
		"Vercel AI Gateway":              {"http://ai-gateway.vercel.sh/v1", "anthropic/claude-sonnet-4.5"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			body, resp := sendToHost(t, tc.baseURL, tc.model,
				llms.WithReasoning(llms.ReasoningNone, 2048), llms.WithMaxTokens(8192))

			if _, ok := body["thinking"]; ok {
				t.Errorf("this host documents no thinking object, got body: %v", body)
			}
			if _, ok := body["reasoning_effort"]; !ok {
				t.Errorf("want the effort the budget maps to, got body: %v", body)
			}
			if w := warningFor(t, resp, "WithReasoning"); w.Asked == w.Sent {
				t.Errorf("the budget that did not travel must be reported, got %+v", w)
			}
		})
	}
}
