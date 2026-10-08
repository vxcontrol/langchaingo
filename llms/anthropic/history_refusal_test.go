package anthropic_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

const (
	boundBlock = "messages.%d.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation. " +
		"Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\". " +
		"That setting requires the `thinking-binding-controls-2026-08-01` value in the `anthropic-beta` header. The system prompt differs."
	brokenSignature = "messages.%d.content.0: Invalid `signature` in `thinking` block"
	modifiedBlocks  = "messages.%d.content.0: `thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified. " +
		"These blocks must remain as they were in the original response."
)

func refusingLLM(t *testing.T, status int, body string) *anthropic.LLM {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-opus-5-5"))
	require.NoError(t, err)
	return llm
}

func refusal(t *testing.T, message string) string {
	t.Helper()

	body, err := json.Marshal(map[string]any{
		"type":       "error",
		"error":      map[string]string{"type": "invalid_request_error", "message": message},
		"request_id": "req_011CSHoEeqs5C35K2UUqR7Fy",
	})
	require.NoError(t, err)
	return string(body)
}

func toolLoop() []llms.MessageContent {
	thought := (&reasoning.ContentReasoning{Content: "plan", Signature: []byte("sig")}).WrittenBy("claude-opus-5-5")
	return []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "rules"),
		llms.TextParts(llms.ChatMessageTypeHuman, "scan the host"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextPartWithReasoning("", thought),
			llms.ToolCall{ID: "call_1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "nmap", Arguments: `{}`}},
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{ToolCallID: "call_1", Name: "nmap", Content: "22/tcp open"}}},
		llms.TextParts(llms.ChatMessageTypeSystem, "report in English"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("port 22 is open", thought)}},
		llms.TextParts(llms.ChatMessageTypeHuman, "go on"),
	}
}

func TestARefusedHistoryNamesTheMessageInTheCallersChain(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name    string
		message string
		sent    int
		fault   llms.HistoryFault
		chain   int
	}{
		{"a changed prefix", boundBlock, 1, llms.HistoryPrefixChanged, 2},
		{"a signature that does not verify", brokenSignature, 3, llms.HistorySignatureInvalid, 5},
		{"modified thinking of the latest answer", modifiedBlocks, 3, llms.HistoryThinkingModified, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			message := fmt.Sprintf(tc.message, tc.sent)
			_, err := refusingLLM(t, http.StatusBadRequest, refusal(t, message)).GenerateContent(t.Context(), toolLoop())

			var rejected *llms.ErrHistoryRejected
			require.ErrorAs(t, err, &rejected)
			require.Equal(t, tc.fault, rejected.Fault)
			require.Equal(t, tc.chain, rejected.Message)
			require.EqualError(t, err, "anthropic: failed to create message: API returned unexpected status code: 400: "+message)
		})
	}
}

func TestATooLargeRequestIsAnOverflow(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name   string
		status int
		body   string
		text   string
	}{
		{
			"a prompt over the context window", http.StatusBadRequest,
			refusal(t, "prompt is too long: 215000 tokens > 200000 maximum"),
			"API returned unexpected status code: 400: prompt is too long: 215000 tokens > 200000 maximum",
		},
		{
			"a request over the size limit", http.StatusRequestEntityTooLarge,
			`{"type":"error","error":{"type":"request_too_large","message":"Request exceeds the maximum allowed number of bytes."}}`,
			"API returned unexpected status code: 413: Request exceeds the maximum allowed number of bytes.",
		},
		{
			"a size limit answered before the API", http.StatusRequestEntityTooLarge,
			"<html><body>413 Request Entity Too Large</body></html>",
			"API returned unexpected status code: 413",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			_, err := refusingLLM(t, tc.status, tc.body).GenerateContent(t.Context(), toolLoop())

			var overflow *llms.ErrContextOverflow
			require.ErrorAs(t, err, &overflow)
			require.EqualError(t, err, "anthropic: failed to create message: "+tc.text)
		})
	}
}

func TestAnotherRefusalKeepsItsPlainError(t *testing.T) {
	t.Parallel()

	_, err := refusingLLM(t, http.StatusBadRequest, refusal(t, "Thinking may not be enabled when tool_choice forces tool use.")).
		GenerateContent(t.Context(), toolLoop())

	var rejected *llms.ErrHistoryRejected
	var overflow *llms.ErrContextOverflow
	require.False(t, errors.As(err, &rejected))
	require.False(t, errors.As(err, &overflow))
	require.EqualError(t, err,
		"anthropic: failed to create message: API returned unexpected status code: 400: Thinking may not be enabled when tool_choice forces tool use.")
}
