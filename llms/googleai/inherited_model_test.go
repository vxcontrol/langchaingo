package googleai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

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

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(server.Close)
	llm, err := New(context.Background(),
		WithAPIKey("unit-test-key"), WithEndpoint(server.URL), WithDefaultModel("gemini-3.1-pro-preview"))
	require.NoError(t, err)
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithReasoningDisabled())
	var refusal *reasoning.ErrReasoningOffUnsupported
	require.ErrorAs(t, err, &refusal, "the listed release keeps its refusal")
}
