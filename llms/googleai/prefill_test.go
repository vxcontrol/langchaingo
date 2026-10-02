package googleai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func sendEndingOnTheModel(t *testing.T, model string) (bool, error) {
	t.Helper()

	sent := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		sent = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{}}`)
	}))
	t.Cleanup(server.Close)

	llm, err := New(context.Background(),
		WithAPIKey("unit-test-key"), WithEndpoint(server.URL), WithDefaultModel(model))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "finish this"),
		llms.TextParts(llms.ChatMessageTypeAI, "the answer is"),
	})
	return sent, err
}

func TestAConversationEndingOnTheModelIsRefusedWhereGeminiRejectsIt(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.8-pro-preview",
		"gemini-4-flash", "gemini-flash-latest", "models/gemini-3.7-flash",
	} {
		sent, err := sendEndingOnTheModel(t, model)
		var refused *reasoning.ErrAssistantPrefillUnsupported
		assert.True(t, errors.As(err, &refused), "%s: got %v", model, err)
		assert.False(t, sent, "%s: the refusal must come before the network", model)
	}

	for _, model := range []string{"gemini-3.5-flash", "gemini-3-pro", "gemini-2.5-flash"} {
		sent, err := sendEndingOnTheModel(t, model)
		require.NoError(t, err, model)
		assert.True(t, sent, model)
	}
}
