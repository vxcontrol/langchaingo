package googleai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func generateForWarnings(t *testing.T, model string, callOpts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{}}`)
	}))
	t.Cleanup(server.Close)

	llm, err := New(context.Background(),
		WithAPIKey("unit-test-key"), WithEndpoint(server.URL), WithDefaultModel(model))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, callOpts...)
	require.NoError(t, err)
	return resp
}

func googleWarningsByOption(warnings []llms.Warning) map[string]llms.Warning {
	byOption := make(map[string]llms.Warning, len(warnings))
	for _, w := range warnings {
		byOption[w.Option] = w
	}
	return byOption
}

func TestOptionsThisDoorNeverReadsAreReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, "gemini-2.5-flash",
		llms.WithN(3), llms.WithLogProbs(true), llms.WithTopLogProbs(2))

	got := googleWarningsByOption(resp.Warnings)
	for option, asked := range map[string]string{
		"WithN": "3", "WithLogProbs": "true", "WithTopLogProbs": "2",
	} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, llms.WarningDrop, w.Kind)
		require.Equal(t, asked, w.Asked)
		require.Equal(t, "gemini-2.5-flash", w.Model)
	}
}

func TestAThinkingBudgetCappedByTheAnswerLimitIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, "gemini-2.5-flash",
		llms.WithMaxTokens(1000), llms.WithReasoning(llms.ReasoningHigh, 8000))

	w, ok := googleWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, w.Kind)
	require.Equal(t, "8000", w.Asked)
	require.Equal(t, "666", w.Sent)
}

func TestAFamilyDrivenByLevelReportsTheLevelItSubstitutes(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, "gemma-4-27b-it",
		llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningLow, 0))

	w, ok := googleWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningSubstitute, w.Kind)
	require.Equal(t, "low", w.Asked)
	require.Equal(t, "HIGH", w.Sent)
}

func TestAPlainGoogleCallCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, "gemini-2.5-flash", llms.WithTemperature(0.2))
	require.Empty(t, resp.Warnings)
}
