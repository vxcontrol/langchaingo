package googleai

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func sentSignatures(t *testing.T, model string, history []llms.MessageContent) []string {
	t.Helper()

	rt := &captureTransport{resp: `{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP","index":0}]}`}
	llm, err := New(t.Context(), WithAPIKey("test-api-key"), WithDefaultModel(model),
		WithHTTPClient(&http.Client{Transport: rt}))
	require.NoError(t, err)
	_, err = llm.GenerateContent(t.Context(), history)
	require.NoError(t, err)

	var payload struct {
		Contents []struct {
			Parts []struct {
				FunctionCall     *struct{} `json:"functionCall"`
				ThoughtSignature string    `json:"thoughtSignature"`
			} `json:"parts"`
		} `json:"contents"`
	}
	require.NoError(t, json.Unmarshal(rt.body, &payload))
	var signatures []string
	for _, content := range payload.Contents {
		for _, part := range content.Parts {
			if part.FunctionCall != nil {
				signatures = append(signatures, part.ThoughtSignature)
			}
		}
	}
	return signatures
}

func toolTurn(id string, signature []byte) llms.MessageContent {
	call := llms.ToolCall{ID: id, Type: "function", FunctionCall: &llms.FunctionCall{Name: "f", Arguments: "{}"}}
	if signature != nil {
		call.Reasoning = &reasoning.ContentReasoning{Signature: signature}
	}
	return llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{call}}
}

func toolResult(id string) llms.MessageContent {
	return llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
		llms.ToolCallResponse{ToolCallID: id, Name: "f", Content: "done"},
	}}
}

func TestGemini3SignsACurrentTurnCallThatArrivedWithoutASignature(t *testing.T) {
	t.Parallel()

	placeholder := base64.StdEncoding.EncodeToString([]byte(geminiSignaturePlaceholder))
	own := []byte("gemini-signature")
	history := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "earlier question"),
		toolTurn("a", nil),
		toolResult("a"),
		llms.TextParts(llms.ChatMessageTypeHuman, "current question"),
		toolTurn("b", nil),
		toolResult("b"),
		toolTurn("c", own),
		toolResult("c"),
	}

	got := sentSignatures(t, "gemini-3-flash-preview", history)
	assert.Equal(t, []string{"", placeholder, base64.StdEncoding.EncodeToString(own)}, got,
		"only the current turn is validated, and a signature the model gave stays as it was")

	assert.Equal(t, []string{"", "", base64.StdEncoding.EncodeToString(own)},
		sentSignatures(t, "gemini-2.5-flash", history), "Gemini 2.5 keeps the history as it came")
}

func TestGemini3SignsOnlyTheFirstOfParallelCalls(t *testing.T) {
	t.Parallel()

	parallel := llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
		llms.ToolCall{ID: "a", Type: "function", FunctionCall: &llms.FunctionCall{Name: "f", Arguments: "{}"}},
		llms.ToolCall{ID: "b", Type: "function", FunctionCall: &llms.FunctionCall{Name: "f", Arguments: "{}"}},
	}}
	history := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "question"),
		parallel,
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "a", Name: "f", Content: "1"},
			llms.ToolCallResponse{ToolCallID: "b", Name: "f", Content: "2"},
		}},
	}

	placeholder := base64.StdEncoding.EncodeToString([]byte(geminiSignaturePlaceholder))
	assert.Equal(t, []string{placeholder, ""}, sentSignatures(t, "gemini-3.1-pro-preview", history))
}
