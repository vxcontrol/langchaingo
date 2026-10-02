package googleai

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestAnUnsignedThoughtWithoutTextIsNotSentAsAnEmptyPart(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gemini-2.5-flash", "gemini-3.8-flash"} {
		rt := &captureTransport{resp: `{"candidates":[{"content":{"role":"model","parts":[{"text":"bye"}]},` +
			`"finishReason":"STOP"}],"usageMetadata":{}}`}
		llm, err := New(t.Context(), WithAPIKey("unit-test-key"), WithDefaultModel(model),
			WithHTTPClient(&http.Client{Transport: rt}))
		require.NoError(t, err)

		_, err = llm.GenerateContent(t.Context(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Say hi."),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.TextPartWithReasoning("", &reasoning.ContentReasoning{Content: "The user wants a greeting."}),
			}},
			llms.TextParts(llms.ChatMessageTypeHuman, "Now say bye."),
		})
		require.NoError(t, err, model)

		var sent struct {
			Contents []struct {
				Role  string           `json:"role"`
				Parts []map[string]any `json:"parts"`
			} `json:"contents"`
		}
		require.NoError(t, json.Unmarshal(rt.body, &sent), model)
		require.Len(t, sent.Contents, 3, model)
		assert.Equal(t, "model", sent.Contents[1].Role, model)
		assert.Empty(t, sent.Contents[1].Parts, "%s: the thought has nothing Gemini accepts back", model)
	}
}
