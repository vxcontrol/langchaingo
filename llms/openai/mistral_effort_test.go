package openai

import (
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestMistralReasoningModelsTakeOnlyTheHighEffort(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"mistral-small-latest", "mistral-medium-latest", "mistral-medium-3-5", "mistral-large-4", "mistral-large-4-0",
	} {
		for _, effort := range []llms.ReasoningEffort{llms.ReasoningLow, llms.ReasoningMedium, llms.ReasoningXHigh} {
			if body := sendForWire(t, model, llms.WithReasoning(effort, 0)); !strings.Contains(body, `"reasoning_effort":"high"`) {
				t.Errorf("%s at %s: Mistral documents only high, got body: %s", model, effort, body)
			}

			resp := sendForWarnings(t, model, llms.WithReasoning(effort, 0))
			if got := warningFor(t, resp, "WithReasoning"); got.Kind != llms.WarningClamp || got.Sent != "high" {
				t.Errorf("%s at %s: warning = %+v", model, effort, got)
			}
		}
		if body := sendForWire(t, model, llms.WithReasoning(llms.ReasoningHigh, 0)); !strings.Contains(body, `"reasoning_effort":"high"`) {
			t.Errorf("%s: high must reach the wire, got body: %s", model, body)
		}
	}
}

func TestMistralLarge4IsTurnedOffWithTheNoneEffort(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"mistral-large-4", "mistral-large-4-0", "mistral/mistral-large-4"} {
		if body := sendForWire(t, model, llms.WithReasoningDisabled()); !strings.Contains(body, `"reasoning_effort":"none"`) {
			t.Errorf("%s: Mistral documents none as the lowest effort that leaves out the thinking chunk, got body: %s", model, body)
		}
	}
}

func TestMistralLarge4ReportsNoneAsItsLowestThinkingLevelOnlyOnMistral(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		baseURL, model string
		floor          bool
	}{
		{"http://api.mistral.ai/v1", "mistral-large-4", true},
		{gatewayBaseURL, "mistral/mistral-large-4", true},
		{"http://ollama.com/v1", "mistral-large-4", false},
		{gatewayBaseURL, "gpt-5.1", false},
		{gatewayBaseURL, "openai/gpt-5.1", false},
	} {
		body, warnings := hostCall(t, tc.baseURL, tc.model, llms.WithReasoningDisabled())
		if body["reasoning_effort"] != "none" {
			t.Errorf("%s on %s: reasoning_effort = %v, want none", tc.model, tc.baseURL, body["reasoning_effort"])
		}
		w, reported := warnings["WithReasoningDisabled"]
		if reported != tc.floor || reported && (w.Kind != llms.WarningSubstitute || w.Sent != "none") {
			t.Errorf("%s on %s: warning = %+v (reported %v), want a substitute only on Mistral", tc.model, tc.baseURL, w, reported)
		}
	}
}

func TestAMistralNameOnAnotherPublicHostTakesTheEffortAsAsked(t *testing.T) {
	t.Parallel()

	for _, effort := range []llms.ReasoningEffort{llms.ReasoningLow, llms.ReasoningMedium} {
		body, warnings := hostCall(t, "http://ollama.com/v1", "mistral-large-4", llms.WithReasoning(effort, 0))
		if body["reasoning_effort"] != string(effort) {
			t.Errorf("at %s: reasoning_effort = %v, want the asked level on a host that is not Mistral", effort, body["reasoning_effort"])
		}
		if w, clamped := warnings["WithReasoning"]; clamped {
			t.Errorf("at %s: unexpected warning %+v", effort, w)
		}
	}
}

func TestMistralModelsThatDoNotReasonGetNoEffort(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"mistral-large-latest", "ministral-8b-latest", "codestral-latest", "devstral-medium-latest"} {
		for _, opt := range []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithReasoning("", 4096)} {
			if body := sendForWire(t, model, opt); strings.Contains(body, `"reasoning_effort"`) {
				t.Errorf("%s: Mistral names no effort for this model, got body: %s", model, body)
			}

			resp := sendForWarnings(t, model, opt)
			if got := warningFor(t, resp, "WithReasoning"); got.Kind != llms.WarningDrop {
				t.Errorf("%s: warning = %+v", model, got)
			}
		}
	}
}

func TestClaudeBehindTheOpenAIDoorKeepsItsTopEfforts(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-sonnet-5", "anthropic/claude-opus-4-7"} {
		for _, effort := range []llms.ReasoningEffort{llms.ReasoningXHigh, llms.ReasoningMax} {
			body := sendForWire(t, model, llms.WithReasoning(effort, 0))
			if !strings.Contains(body, `"reasoning_effort":"`+string(effort)+`"`) {
				t.Errorf("%s at %s: Anthropic serves this level off Bedrock, got body: %s", model, effort, body)
			}
		}
	}
}
