package reasoning

import "testing"

func TestOnlyModelsThatReasonOnMistralTakeThinkingBackInContent(t *testing.T) {
	t.Parallel()

	for model, want := range map[string]bool{
		"mistral-medium-latest":         true,
		"mistral-small-latest":          true,
		"magistral-small-latest":        true,
		"zai-glm-5-2":                   true,
		"glm-5-2":                       true,
		"mistral/mistral-medium-latest": true,
		"codestral-latest":              false,
		"mistral-large-latest":          false,
		"ministral-8b-latest":           false,
		"mistral/ministral-8b-latest":   false,
		"mistral/codestral-latest":      false,
		"glm-5.2":                       false,
		"deepseek-v4-pro":               false,
		"kimi-k3":                       false,

		"mistralai/mistral-small-2603":            false,
		"mistral-ai/mistral-small-2603":           false,
		"openrouter/mistralai/mistral-medium-3-5": false,
		"mistral-medium-3.5:latest":               false,
		"magistral:24b":                           false,
		"magistral":                               false,
	} {
		if got := ReplaysThinkingInContent(model); got != want {
			t.Errorf("ReplaysThinkingInContent(%q) = %v, want %v", model, got, want)
		}
	}
}
