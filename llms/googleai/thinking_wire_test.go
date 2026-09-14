package googleai

import (
	"context"
	"encoding/json"
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

func thinkingConfigFor(t *testing.T, model string, opts ...llms.CallOption) map[string]any {
	t.Helper()

	var body struct {
		GenerationConfig struct {
			ThinkingConfig map[string]any `json:"thinkingConfig"`
		} `json:"generationConfig"`
	}
	require.NoError(t, json.Unmarshal([]byte(thinkingWireFor(t, model, opts...)), &body))
	return body.GenerationConfig.ThinkingConfig
}

func budgetOnTheWire(t *testing.T, model string, opts ...llms.CallOption) float64 {
	t.Helper()

	tc := thinkingConfigFor(t, model, opts...)
	budget, ok := tc["thinkingBudget"].(float64)
	require.True(t, ok, "no thinkingBudget for %s, got %v", model, tc)
	return budget
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

	for _, model := range []string{"gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"} {
		budget := budgetOnTheWire(t, model, llms.WithAdaptiveReasoning(""), llms.WithMaxTokens(8192))
		assert.Equal(t, float64(-1), budget, "%s takes the dynamic budget sentinel", model)
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
		want  float64
	}{
		{"gemini-2.5-pro", 50, 128},
		{"gemini-2.5-pro", 40000, 32768},
		{"gemini-2.5-flash-lite", 100, 512},
		{"gemini-2.5-flash-lite", 30000, 24576},
		{"gemini-2.5-flash", 1, 1},
		{"gemini-2.5-flash", 30000, 24576},
	} {
		budget := budgetOnTheWire(t, tc.model,
			llms.WithReasoning(llms.ReasoningNone, tc.asked), llms.WithMaxTokens(65536))
		assert.Equal(t, tc.want, budget, "%s asked %d", tc.model, tc.asked)
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

func TestAdaptiveWithANamedEffortStillHonoursThatEffort(t *testing.T) {
	t.Parallel()

	for effort, want := range map[llms.ReasoningEffort]float64{
		llms.ReasoningLow: 2048, llms.ReasoningMedium: 2730, llms.ReasoningHigh: 4096,
	} {
		adaptive := budgetOnTheWire(t, "gemini-2.5-flash",
			llms.WithAdaptiveReasoning(effort), llms.WithMaxTokens(8192))
		named := budgetOnTheWire(t, "gemini-2.5-flash",
			llms.WithReasoning(effort, 0), llms.WithMaxTokens(8192))
		assert.Equal(t, want, adaptive, "%s: a named effort is a depth, not a hand-off", effort)
		assert.Equal(t, named, adaptive, "%s: adaptive must not change the depth the effort names", effort)
	}

	tc := thinkingConfigFor(t, "gemini-3.5-flash",
		llms.WithAdaptiveReasoning(llms.ReasoningMedium), llms.WithMaxTokens(8192))
	assert.Equal(t, "MEDIUM", tc["thinkingLevel"])
	assert.NotContains(t, tc, "thinkingBudget")
}
