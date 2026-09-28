package bedrockclient

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func additionalFields(t *testing.T, in *ConverseInput) map[string]any {
	t.Helper()

	got, err := NewConverseClient(&MockBedrockRuntimeClient{}).buildConverseInput(in)
	require.NoError(t, err)
	if got.AdditionalModelRequestFields == nil {
		return nil
	}
	raw, err := got.AdditionalModelRequestFields.MarshalSmithyDocument()
	require.NoError(t, err)
	var fields map[string]any
	require.NoError(t, json.Unmarshal(raw, &fields))
	return fields
}

func humanTurn() []Message {
	return []Message{{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"}}
}

func TestConverseCarriesTheTopKTheCallerAskedFor(t *testing.T) {
	t.Parallel()

	topK := 40
	fields := additionalFields(t, &ConverseInput{
		ModelID:  "us.anthropic.claude-haiku-4-5-20251001-v1:0",
		Messages: humanTurn(),
		TopK:     &topK,
	})

	require.NotNil(t, fields, "a caller that named top_k gets additional model fields")
	assert.EqualValues(t, 40, fields["top_k"],
		"the vendor takes top_k beside thinking, and the legacy door already sends it")
}

func TestConverseDropsTopKWhileThinking(t *testing.T) {
	t.Parallel()

	topK, budget := 40, 2048
	fields := additionalFields(t, &ConverseInput{
		ModelID:         "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		Messages:        humanTurn(),
		TopK:            &topK,
		MaxTokens:       &[]int{8192}[0],
		ReasoningConfig: &llms.ReasoningConfig{Tokens: budget},
	})

	require.NotNil(t, fields)
	assert.Contains(t, fields, "thinking", "the request is a thinking one")
	assert.NotContains(t, fields, "top_k",
		"budget thinking rejects top_k, exactly as the legacy door already handles it")
}

func TestConverseSendsNoAdditionalFieldsWhenTheCallerNamedNothing(t *testing.T) {
	t.Parallel()

	fields := additionalFields(t, &ConverseInput{
		ModelID:  "us.anthropic.claude-haiku-4-5-20251001-v1:0",
		Messages: humanTurn(),
	})

	assert.Nil(t, fields, "a bare request must not grow a field the caller never named")
}

func TestConverseKeepsTheNovaShapeWhenTopKIsNamed(t *testing.T) {
	t.Parallel()

	topK := 40
	fields := additionalFields(t, &ConverseInput{
		ModelID:         "us.amazon.nova-2-lite-v1:0",
		Messages:        humanTurn(),
		TopK:            &topK,
		ReasoningConfig: &llms.ReasoningConfig{Effort: llms.ReasoningHigh},
	})

	require.NotNil(t, fields)
	assert.Contains(t, fields, "reasoningConfig", "nova keeps its own shape")
	assert.NotContains(t, fields, "top_k", "top_k is an anthropic field, not a nova one")
}

func TestConverseSendsNoTopKToAFamilyThatDoesNotDocumentIt(t *testing.T) {
	t.Parallel()

	topK := 40
	fields := additionalFields(t, &ConverseInput{
		ModelID:  "meta.llama3-70b-instruct-v1:0",
		Messages: humanTurn(),
		TopK:     &topK,
	})

	assert.Nil(t, fields,
		"top_k in additionalModelRequestFields is an anthropic field; inventing it elsewhere is a guess")
}
