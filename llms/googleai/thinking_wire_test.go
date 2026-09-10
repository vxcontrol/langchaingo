package googleai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func thinkingWireFor(t *testing.T, model string, opts ...llms.CallOption) string {
	t.Helper()

	var body string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{}}`)
	}))
	t.Cleanup(server.Close)

	llm, err := New(context.Background(),
		WithAPIKey("unit-test-key"), WithEndpoint(server.URL), WithDefaultModel(model))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	require.NoError(t, err)
	return body
}

func TestNoThinkingBudgetReachesAGemini3Model(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
		"gemini-3.1-flash-lite", "gemini-3-flash-preview",
	} {
		for name, opt := range map[string]llms.CallOption{
			"disabled": llms.WithReasoningDisabled(),
			"effort":   llms.WithReasoning(llms.ReasoningMedium, 0),
		} {
			body := thinkingWireFor(t, model, opt, llms.WithMaxTokens(8192))
			assert.NotContains(t, body, "thinkingBudget",
				"%s on %s: this generation is driven by the level scale", name, model)
		}
	}
}

func TestDisablingThinkingOnAGemini3ModelSendsTheLowestLevel(t *testing.T) {
	t.Parallel()

	body := thinkingWireFor(t, "gemini-3.5-flash", llms.WithReasoningDisabled())
	assert.Contains(t, body, `"thinkingLevel":"MINIMAL"`)
}

func TestDisablingThinkingOnGemini25StillSendsBudgetZero(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gemini-2.5-flash", "gemini-2.5-flash-lite"} {
		body := thinkingWireFor(t, model, llms.WithReasoningDisabled())
		assert.Contains(t, body, `"thinkingBudget":0`, "%s takes the budget wire", model)
		assert.NotContains(t, body, "thinkingLevel",
			"%s answers with an error when a level is present", model)
	}
}

func TestAdaptiveThinkingLetsTheModelChoose(t *testing.T) {
	t.Parallel()

	for model, want := range map[string]string{
		"gemini-2.5-flash":      `"thinkingBudget":-1`,
		"gemini-2.5-flash-lite": `"thinkingBudget":-1`,
		"gemini-2.5-pro":        `"thinkingBudget":-1`,
	} {
		body := thinkingWireFor(t, model, llms.WithAdaptiveReasoning(""), llms.WithMaxTokens(8192))
		assert.Contains(t, body, want, "%s takes the dynamic budget sentinel", model)
	}

	for _, model := range []string{"gemini-3.5-flash", "gemini-3-flash-preview"} {
		body := thinkingWireFor(t, model, llms.WithAdaptiveReasoning(""), llms.WithMaxTokens(8192))
		assert.NotContains(t, body, "thinkingLevel", "%s decides its own depth", model)
		assert.NotContains(t, body, "thinkingBudget", "%s reads no budget", model)
		assert.Contains(t, body, `"includeThoughts":true`, "%s still returns its thought summaries", model)
	}
}

func TestABudgetOutsideTheModelRangeIsHeldInside(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model string
		asked int
		want  string
	}{
		{"gemini-2.5-pro", 50, `"thinkingBudget":128`},
		{"gemini-2.5-flash-lite", 100, `"thinkingBudget":512`},
		{"gemini-2.5-flash", 1, `"thinkingBudget":1`},
	} {
		body := thinkingWireFor(t, tc.model,
			llms.WithReasoning(llms.ReasoningNone, tc.asked), llms.WithMaxTokens(65536))
		assert.Contains(t, body, tc.want, "%s asked %d", tc.model, tc.asked)
	}
}

func TestAModelThatDoesNotThinkGetsNoThinkingConfig(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gemini-2.0-flash", "gemini-1.5-pro", "gemma-3-27b-it"} {
		for name, opt := range map[string]llms.CallOption{
			"adaptive": llms.WithAdaptiveReasoning(""),
			"effort":   llms.WithReasoning(llms.ReasoningHigh, 0),
			"budget":   llms.WithReasoning(llms.ReasoningNone, 4096),
			"disabled": llms.WithReasoningDisabled(),
		} {
			body := thinkingWireFor(t, model, opt, llms.WithMaxTokens(16384))
			assert.NotContains(t, body, "thinkingConfig",
				"%s on %s: this family takes no thinking control", name, model)
		}
	}
}
