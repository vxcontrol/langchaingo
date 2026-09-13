package openai

import (
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestMistralReasoningModelsTakeOnlyTheHighEffort(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"mistral-small-latest", "mistral-medium-latest", "mistral-medium-3-5"} {
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
