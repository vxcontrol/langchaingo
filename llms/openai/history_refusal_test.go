package openai

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func gatewayRefusing(t *testing.T, status int, body string) error {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)

	_, err := newUnitLLM(t, WithBaseURL(srv.URL), WithModel("anthropic/claude-opus-5-5")).
		GenerateContent(t.Context(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeSystem, "rules"),
			llms.TextParts(llms.ChatMessageTypeHuman, "scan the host"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.ToolCall{ID: "call_1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "nmap", Arguments: `{}`}},
			}},
			{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{ToolCallID: "call_1", Name: "nmap", Content: "22/tcp open"}}},
			llms.TextParts(llms.ChatMessageTypeHuman, "go on"),
		})
	return err
}

func litellmWrapped(t *testing.T, anthropicMessage string) (body, message string) {
	t.Helper()

	type apiError struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	}
	inner := jsonAsSent(t, struct {
		Type      string   `json:"type"`
		Error     apiError `json:"error"`
		RequestID string   `json:"request_id"`
	}{"error", apiError{"invalid_request_error", anthropicMessage}, "req_011CePEbLTUpNsaSgA74pmPX"})
	message = "litellm.BadRequestError: AnthropicException - " + inner +
		". Received Model Group=anthropic/claude-opus-5-5\nAvailable Model Group Fallbacks=None"
	return jsonAsSent(t, map[string]any{"error": map[string]any{"message": message, "type": nil, "param": nil, "code": "400"}}), message
}

func jsonAsSent(t *testing.T, v any) string {
	t.Helper()

	var buf strings.Builder
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	require.NoError(t, enc.Encode(v))
	return strings.TrimSuffix(buf.String(), "\n")
}

func TestAGatewaysHistoryRefusalIsTypedWithoutAPosition(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		anthropic string
		fault     llms.HistoryFault
	}{
		{
			"messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation. " +
				"Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\".",
			llms.HistoryPrefixChanged,
		},
		{"messages.3.content.0: Invalid `signature` in `thinking` block", llms.HistorySignatureInvalid},
		{
			"messages.3.content.0: `thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified. " +
				"These blocks must remain as they were in the original response.",
			llms.HistoryThinkingModified,
		},
	} {
		body, message := litellmWrapped(t, tc.anthropic)
		err := gatewayRefusing(t, http.StatusBadRequest, body)

		var rejected *llms.ErrHistoryRejected
		require.ErrorAs(t, err, &rejected, tc.fault)
		require.Equal(t, tc.fault, rejected.Fault)
		require.Equal(t, -1, rejected.Message, "the gateway built the messages Claude names")
		require.EqualError(t, err, "API returned unexpected status code: 400: "+message)
	}
}

func TestAGatewaysTooLargeRequestIsAnOverflow(t *testing.T) {
	t.Parallel()

	body, message := litellmWrapped(t, "prompt is too long: 215000 tokens > 200000 maximum")
	err := gatewayRefusing(t, http.StatusBadRequest, body)
	var overflow *llms.ErrContextOverflow
	require.ErrorAs(t, err, &overflow)
	require.EqualError(t, err, "API returned unexpected status code: 400: "+message)

	err = gatewayRefusing(t, http.StatusRequestEntityTooLarge, "<html><body>413 Request Entity Too Large</body></html>")
	require.ErrorAs(t, err, &overflow)
	require.EqualError(t, err, "API returned unexpected status code: 413: <html><body>413 Request Entity Too Large</body></html>")
}

func TestAGatewaysOtherRefusalKeepsItsPlainError(t *testing.T) {
	t.Parallel()

	body, message := litellmWrapped(t, "Thinking may not be enabled when tool_choice forces tool use.")
	err := gatewayRefusing(t, http.StatusBadRequest, body)

	var rejected *llms.ErrHistoryRejected
	var overflow *llms.ErrContextOverflow
	require.False(t, errors.As(err, &rejected))
	require.False(t, errors.As(err, &overflow))
	require.EqualError(t, err, "API returned unexpected status code: 400: "+message)
}
