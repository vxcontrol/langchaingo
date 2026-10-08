package googleai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

const signedGemini = "gemini-3-pro-preview"

func replayedCallSignature(t *testing.T, signature *reasoning.ContentReasoning) string {
	t.Helper()

	server, body := recordingServer(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},`+
		`"finishReason":"STOP"}],"usageMetadata":{}}`)
	llm, err := New(context.Background(), WithAPIKey("unit-test-key"), WithEndpoint(server.URL), WithDefaultModel(signedGemini))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "look it up"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{
			ID: "c1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: `{}`}, Reasoning: signature,
		}}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "c1", Name: "lookup", Content: "done"},
		}},
	})
	require.NoError(t, err)

	var sent struct {
		Contents []struct {
			Parts []map[string]any `json:"parts"`
		} `json:"contents"`
	}
	require.NoError(t, json.Unmarshal(*body, &sent))
	signatureSent, _ := sent.Contents[1].Parts[0]["thoughtSignature"].(string)
	return signatureSent
}

func TestGeminiGetsBackOnlyTheThoughtSignaturesItCanVerify(t *testing.T) {
	t.Parallel()

	signedBy := func(model string) *reasoning.ContentReasoning {
		return (&reasoning.ContentReasoning{Signature: []byte("sig")}).WrittenBy(model)
	}
	encoded := func(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }

	require.Equal(t, encoded("sig"), replayedCallSignature(t, signedBy("gemini-2.5-pro")))
	require.Equal(t, encoded("sig"), replayedCallSignature(t, signedBy("")), "written before the writer was recorded")
	require.Equal(t, encoded(geminiSignaturePlaceholder), replayedCallSignature(t, signedBy("claude-sonnet-4-5")),
		"a call of the current turn that Claude signed takes the documented placeholder")
}

func TestAGeminiAnswersThoughtSignatureNamesTheModelThatWroteIt(t *testing.T) {
	t.Parallel()

	server, _ := recordingServer(t, `{"candidates":[{"content":{"role":"model","parts":[`+
		`{"functionCall":{"name":"lookup","args":{}},"thoughtSignature":"c2ln"}]},"finishReason":"STOP"}],"usageMetadata":{}}`)
	llm, err := New(context.Background(), WithAPIKey("unit-test-key"), WithEndpoint(server.URL), WithDefaultModel(signedGemini))
	require.NoError(t, err)

	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}})
	for _, conversation := range [][]llms.MessageContent{
		{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		{llms.TextParts(llms.ChatMessageTypeSystem, "be brief"), llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
	} {
		resp, err := llm.GenerateContent(context.Background(), conversation, tools)
		require.NoError(t, err)
		require.Len(t, resp.Choices[0].ToolCalls, 1)
		require.Equal(t, signedGemini, resp.Choices[0].ToolCalls[0].Reasoning.Model, "%d messages", len(conversation))
	}
}
