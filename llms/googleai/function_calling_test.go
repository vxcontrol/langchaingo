package googleai

import (
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestToolsAreRefusedBeforeTheNetworkForAGeminiWithoutFunctionCalling(t *testing.T) {
	t.Parallel()

	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}})
	send := func(model string) (*captureTransport, error) {
		rt := &captureTransport{resp: `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},` +
			`"finishReason":"STOP"}],"usageMetadata":{}}`}
		llm, err := New(t.Context(), WithAPIKey("unit-test-key"), WithDefaultModel(model),
			WithHTTPClient(&http.Client{Transport: rt}))
		require.NoError(t, err)
		_, err = llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, tools)
		return rt, err
	}

	for _, model := range []string{"gemini-3.8-flash-cyber", "models/gemini-3.8-flash-cyber"} {
		rt, err := send(model)
		var unsupported *reasoning.ErrFunctionCallingUnsupported
		require.True(t, errors.As(err, &unsupported), "%s: %v", model, err)
		assert.Nil(t, rt.body, "%s: refused before the network", model)
	}

	rt, err := send("gemini-3.8-flash")
	require.NoError(t, err)
	assert.NotNil(t, rt.body)
}
