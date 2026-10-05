package googleai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestAnUnlistedGeminiIsSentTheThinkingItsReleaseTakes(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gemini-4-flash", "gemini-4-pro"} {
		config, warnings := generationConfigSent(t, model, llms.WithReasoning(llms.ReasoningHigh, 0))
		require.Equal(t, map[string]any{"includeThoughts": true, "thinkingLevel": "HIGH"}, config["thinkingConfig"], model)
		require.Len(t, warnings, 1, "%s: only the inherited model is reported: %v", model, warnings)
		require.Equal(t, llms.WarningInherit, warnings["WithModel"].Kind, model)
	}
}

func TestAnUnlistedGeminiIsSentADisableItsReleaseWouldRefuse(t *testing.T) {
	t.Parallel()

	config, _ := generationConfigSent(t, "gemini-4-flash", llms.WithReasoningDisabled())
	thinking, _ := config["thinkingConfig"].(map[string]any)
	require.Equal(t, "MINIMAL", thinking["thinkingLevel"], "the newest flash that documents a disable: %v", config)
	resp := generateForWarnings(t, "gemini-4-flash", llms.WithReasoningDisabled())
	inherited := map[string]bool{}
	for _, w := range resp.Warnings {
		if w.Kind == llms.WarningInherit {
			inherited[w.Option] = true
		}
	}
	require.Equal(t, map[string]bool{"WithModel": true, "WithReasoningDisabled": true}, inherited)
	for _, w := range resp.Warnings {
		if w.Option == "WithReasoningDisabled" {
			require.Equal(t, "minimal", w.Sent, "%v", w)
		}
	}

	config, _ = generationConfigSent(t, "gemini-4-pro", llms.WithReasoningDisabled())
	require.NotContains(t, config, "thinkingConfig", "no pro release documents a disable: %v", config)
	inherited = map[string]bool{}
	for _, w := range generateForWarnings(t, "gemini-4-pro", llms.WithReasoningDisabled()).Warnings {
		if w.Kind == llms.WarningInherit {
			inherited[w.Option] = true
		}
		if w.Option == "WithReasoningDisabled" {
			require.Empty(t, w.Sent, "%v", w)
		}
	}
	require.Equal(t, map[string]bool{"WithModel": true, "WithReasoningDisabled": true}, inherited,
		"the disable a pro cannot honour is reported, not lost")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(server.Close)
	for _, model := range []string{"gemini-3.1-pro-preview", "gemini-3.1-pro", "gemini-pro-latest"} {
		llm, err := New(context.Background(),
			WithAPIKey("unit-test-key"), WithEndpoint(server.URL), WithDefaultModel(model))
		require.NoError(t, err)
		_, err = llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithReasoningDisabled())
		var refusal *reasoning.ErrReasoningOffUnsupported
		require.ErrorAs(t, err, &refusal, "%s: the listed release keeps its refusal", model)
	}

	for _, w := range generateForWarnings(t, "gemini-3.1-flash-lite").Warnings {
		require.NotEqual(t, llms.WarningInherit, w.Kind, "the tables list gemini-3.1-flash-lite: %v", w)
	}
}

func TestAClaudeWordInTheProjectIDChangesNothingSent(t *testing.T) {
	t.Parallel()

	const path = "projects/%s/locations/us-central1/publishers/google/models/"
	for _, model := range []string{"gemini-4-pro", "gemini-4-flash"} {
		for _, opt := range []llms.CallOption{llms.WithReasoningDisabled(), llms.WithReasoning(llms.ReasoningHigh, 0)} {
			neutral, prefixed := fmt.Sprintf(path, "relay-team")+model, fmt.Sprintf(path, "claude-team")+model
			want, wantWarnings := generationConfigSent(t, neutral, opt)
			got, gotWarnings := generationConfigSent(t, prefixed, opt)
			require.Equal(t, want, got, prefixed)
			require.Equal(t, warningsAbout(neutral, wantWarnings), warningsAbout(prefixed, gotWarnings), prefixed)
		}
	}
}

func warningsAbout(model string, byOption map[string]llms.Warning) map[string]llms.Warning {
	named := strings.NewReplacer(model, "<model>")
	normalized := make(map[string]llms.Warning, len(byOption))
	for option, w := range byOption {
		w.Model, w.Asked, w.Sent, w.Reason = "", named.Replace(w.Asked), named.Replace(w.Sent), named.Replace(w.Reason)
		normalized[option] = w
	}
	return normalized
}
