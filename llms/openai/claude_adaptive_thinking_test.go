package openai

import (
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestDelegatedAdaptiveTurnsThinkingOnWhereClaudeWaitsToBeAsked(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"claude-opus-4-8", "anthropic/claude-opus-4-8", "anthropic/claude-opus-4-7",
		"anthropic/claude-opus-4-6", "anthropic/claude-sonnet-4-6",
		"us.anthropic.claude-opus-4-8", "bedrock/us.anthropic.claude-opus-4-6-v1",
		"bedrock/global.anthropic.claude-sonnet-4-6", "vertex_ai/claude-opus-4-7",
	} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			body := captureWireWithClient(t, model, nil, llms.WithAdaptiveReasoning(llms.ReasoningNone))

			if !strings.Contains(body, `"thinking":{"type":"adaptive"`) {
				t.Errorf("thinking stays off on this model until adaptive is set, got body: %s", body)
			}
			if strings.Contains(body, "reasoning_effort") || strings.Contains(body, "budget_tokens") {
				t.Errorf("the caller named no depth, got body: %s", body)
			}
		})
	}
}

func TestDelegatedAdaptiveIsReportedWhereTheHostDocumentsNoThinkingObject(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct{ baseURL, model string }{
		"the gateway's deepinfra route":   {gatewayBaseURL, "deepinfra/anthropic/claude-opus-4-8"},
		"the gateway's perplexity route":  {gatewayBaseURL, "perplexity/anthropic/claude-sonnet-4-6"},
		"the gateway's openrouter route":  {gatewayBaseURL, "openrouter/anthropic/claude-opus-4.8"},
		"the gateway's azure_ai route":    {gatewayBaseURL, "azure_ai/claude-opus-4-8"},
		"the gateway's together_ai route": {gatewayBaseURL, "together_ai/anthropic/claude-opus-4-8"},
		"DeepInfra":                       {"http://api.deepinfra.com/v1/openai", "anthropic/claude-opus-4-8"},
		"Perplexity":                      {"http://api.perplexity.ai", "anthropic/claude-opus-4-7"},
		"OpenRouter":                      {"http://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.6"},
		"ZenMux":                          {"http://zenmux.ai/api/v1", "anthropic/claude-opus-4.8"},
		"Vercel AI Gateway":               {"http://ai-gateway.vercel.sh/v1", "anthropic/claude-opus-4.8"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			body, resp := sendToHost(t, tc.baseURL, tc.model, llms.WithAdaptiveReasoning(llms.ReasoningNone))

			if _, ok := body["thinking"]; ok {
				t.Errorf("this host documents no thinking object, got body: %v", body)
			}
			if w := warningFor(t, resp, "WithAdaptiveReasoning"); w.Kind != llms.WarningDrop || w.Sent != "" {
				t.Errorf("the lost adaptive request must be reported, got %+v", w)
			}
		})
	}
}

func TestDelegatedAdaptiveIsReportedOnAnthropicsOpenAICompatibilityEndpoint(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			body, resp := sendToHost(t, "http://api.anthropic.com/v1", model, llms.WithAdaptiveReasoning(llms.ReasoningNone))

			if _, ok := body["thinking"]; ok {
				t.Errorf("this endpoint answers adaptive thinking with a 400, got body: %v", body)
			}
			w := warningFor(t, resp, "WithAdaptiveReasoning")
			if w.Kind != llms.WarningDrop || w.Sent != "" || !strings.Contains(w.Reason, "refuses adaptive thinking") {
				t.Errorf("the refused adaptive request must be reported as refused, got %+v", w)
			}
		})
	}
}

func TestAClaudeRequestWithoutReasoningStaysWithoutThinking(t *testing.T) {
	t.Parallel()

	for model, keepsTemperature := range map[string]bool{
		"anthropic/claude-opus-4-8":               false,
		"anthropic/claude-sonnet-4-6":             true,
		"bedrock/us.anthropic.claude-opus-4-6-v1": true,
	} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			_, sent := sendForWarningsWith(t, model, nil, llms.WithTemperature(0.3))

			if _, ok := sent["thinking"]; ok {
				t.Errorf("the caller asked for no thinking, got %v", sent)
			}
			if keepsTemperature && sent["temperature"] != 0.3 {
				t.Errorf("a request without thinking keeps the caller's temperature, got %v", sent)
			}
		})
	}
}

func TestDelegatedAdaptiveAsksForTheThinkingTextTheAnthropicDoorReturns(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"anthropic/claude-opus-4-8", "anthropic/claude-opus-4-7", "anthropic/claude-sonnet-4-6"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			fields := extraBodyWire(t, model, nil, "ok", llms.WithAdaptiveReasoning(llms.ReasoningNone))
			assertWireJSON(t, fields, map[string]string{"thinking": `{"type":"adaptive","display":"summarized"}`})
		})
	}
}

func TestDelegatedAdaptiveThinkingTakesTheSamplingThinkingRefuses(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-6"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			resp, sent := sendForWarningsWith(t, model, nil, llms.WithAdaptiveReasoning(llms.ReasoningNone),
				llms.WithTemperature(0.3), llms.WithTopK(40))

			if _, ok := sent["thinking"]; !ok {
				t.Fatalf("want the adaptive thinking object, got %v", sent)
			}
			if temperature, ok := sent["temperature"]; ok && temperature != 1.0 {
				t.Errorf("thinking takes temperature 1 or none, got %v", temperature)
			}
			if _, ok := sent["top_k"]; ok {
				t.Errorf("thinking takes no top_k, got %v", sent)
			}
			if w := warningFor(t, resp, "WithTemperature"); w.Asked != "0.3" {
				t.Errorf("the lost temperature must be reported, got %+v", w)
			}
		})
	}
}

func TestDelegatedAdaptiveReachingTheWireIsNotReportedAsLost(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "anthropic/claude-opus-4-8",
		llms.WithAdaptiveReasoning(llms.ReasoningNone), llms.WithTemperature(0.3), llms.WithTopP(0.9))

	for _, w := range resp.Warnings {
		if w.Option != "WithTemperature" && w.Option != "WithTopP" {
			t.Errorf("only the sampling the model rejects was lost, got %+v", w)
		}
	}
}

func TestDelegatedAdaptiveKeepsItsObjectOffOtherWires(t *testing.T) {
	t.Parallel()

	offAndAdaptive := func(o *llms.CallOptions) {
		o.Reasoning = &llms.ReasoningConfig{Mode: llms.ReasoningOff, Adaptive: true}
	}
	for name, tc := range map[string]struct {
		model  string
		client []Option
		opts   []llms.CallOption
	}{
		"a generation that predates adaptive": {
			model: "anthropic/claude-haiku-4-5", opts: []llms.CallOption{llms.WithAdaptiveReasoning(llms.ReasoningNone)},
		},
		"a model that is not Claude": {
			model: "glm-5.2", opts: []llms.CallOption{llms.WithAdaptiveReasoning(llms.ReasoningNone)},
		},
		"the reasoning object format": {
			model: "anthropic/claude-opus-4-8", client: []Option{WithModernReasoningFormat()},
			opts: []llms.CallOption{llms.WithAdaptiveReasoning(llms.ReasoningNone)},
		},
		"an explicit off": {
			model: "anthropic/claude-opus-4-8", opts: []llms.CallOption{offAndAdaptive},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			body := captureWireWithClient(t, tc.model, tc.client, tc.opts...)
			if strings.Contains(body, `"adaptive"`) {
				t.Errorf("got body: %s", body)
			}
		})
	}
}

func TestExtraBodyThinkingMeetsTheDelegatedAdaptiveObject(t *testing.T) {
	t.Parallel()

	extra := func(thinking map[string]any) llms.CallOption {
		return llms.WithExtraBody(map[string]any{"thinking": thinking})
	}
	for name, tc := range map[string]struct {
		thinking map[string]any
		want     string
	}{
		"a key of the same object merges":  {map[string]any{"display": "omitted"}, `{"type":"adaptive","display":"omitted"}`},
		"another type replaces the object": {map[string]any{"type": "disabled"}, `{"type":"disabled"}`},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			fields := extraBodyWire(t, "anthropic/claude-opus-4-8", nil, "ok",
				llms.WithAdaptiveReasoning(llms.ReasoningNone), extra(tc.thinking))
			assertWireJSON(t, fields, map[string]string{"thinking": tc.want})
		})
	}
}

func TestDelegatedAdaptiveLeavesTheWireEmptyWhereClaudeThinksAlready(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"anthropic/claude-opus-5", "anthropic/claude-sonnet-5", "anthropic/claude-fable-5",
		"anthropic/claude-fable-5-1", "anthropic/claude-mythos-5", "anthropic/claude-mythos-5-1",
		"claude-mythos-preview", "bedrock/us.anthropic.claude-opus-5",
	} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			body := captureWireWithClient(t, model, nil, llms.WithAdaptiveReasoning(llms.ReasoningNone))

			if strings.Contains(body, `"thinking"`) || strings.Contains(body, "reasoning_effort") {
				t.Errorf("this model thinks with no configuration, got body: %s", body)
			}
		})
	}
}
