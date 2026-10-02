package googleai

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func sendConversation(
	t *testing.T, defaultModel string, messages []llms.MessageContent, opts ...llms.CallOption,
) ([]string, error) {
	t.Helper()

	rt := &captureTransport{resp: `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},` +
		`"finishReason":"STOP"}],"usageMetadata":{}}`}
	llm, err := New(t.Context(), WithAPIKey("unit-test-key"), WithDefaultModel(defaultModel),
		WithHTTPClient(&http.Client{Transport: rt}))
	require.NoError(t, err)

	_, err = llm.GenerateContent(t.Context(), messages, opts...)
	if rt.body == nil {
		return nil, err
	}
	var sent struct {
		Contents []struct {
			Role string `json:"role"`
		} `json:"contents"`
	}
	require.NoError(t, json.Unmarshal(rt.body, &sent))
	roles := make([]string, 0, len(sent.Contents))
	for _, content := range sent.Contents {
		roles = append(roles, content.Role)
	}
	return roles, err
}

func TestAConversationEndingOnTheModelIsRefusedWhereGeminiRejectsIt(t *testing.T) {
	t.Parallel()

	human := llms.TextParts(llms.ChatMessageTypeHuman, "finish this")
	answer := llms.TextParts(llms.ChatMessageTypeAI, "the answer is")
	endingOnTheModel := []llms.MessageContent{human, answer}

	refused := func(t *testing.T, label string, roles []string, err error) {
		t.Helper()
		var refusal *reasoning.ErrAssistantPrefillUnsupported
		assert.True(t, errors.As(err, &refusal), "%s: got %v", label, err)
		assert.Nil(t, roles, "%s: the refusal must come before the network", label)
	}

	for _, model := range []string{
		"gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.8-pro-preview",
		"gemini-4-flash", "models/gemini-3.7-flash",
	} {
		roles, err := sendConversation(t, model, endingOnTheModel)
		refused(t, model, roles, err)
	}

	roles, err := sendConversation(t, "gemini-2.5-flash", endingOnTheModel, llms.WithModel("gemini-3.7-flash"))
	refused(t, "a per-call model over the client's default", roles, err)

	for name, messages := range map[string][]llms.MessageContent{
		"a system message after the answer": {human, answer, llms.TextParts(llms.ChatMessageTypeSystem, "be brief")},
		"an empty turn after the answer":    {human, answer, llms.TextParts(llms.ChatMessageTypeHuman, "")},
	} {
		roles, err := sendConversation(t, "gemini-3.8-flash", messages)
		refused(t, name, roles, err)
	}

	for _, model := range []string{
		"gemini-3.5-flash", "gemini-3-pro", "gemini-2.5-flash", "gemini-flash-latest", "gemini-pro-latest",
		"gemini-2.5-flash-native-audio-latest",
	} {
		roles, err := sendConversation(t, model, endingOnTheModel)
		require.NoError(t, err, model)
		assert.Equal(t, []string{"user", "model"}, roles, model)
	}

	roles, err = sendConversation(t, "gemini-3.8-flash",
		[]llms.MessageContent{human, llms.TextParts(llms.ChatMessageTypeAI, "")})
	require.NoError(t, err, "a model turn with nothing in it is not the last non-empty turn")
	assert.Equal(t, []string{"user", "model"}, roles)
}
