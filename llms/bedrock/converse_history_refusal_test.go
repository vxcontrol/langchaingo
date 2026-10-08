package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/smithy-go"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func converseRefusal(t *testing.T, message string, streamed bool) error {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		body, err := json.Marshal(map[string]string{"message": message})
		require.NoError(t, err)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Amzn-Errortype", "ValidationException")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	thought := (&reasoning.ContentReasoning{Content: "plan", Signature: []byte("sig")}).WrittenBy(converseClaude)
	var opts []llms.CallOption
	if streamed {
		opts = append(opts, llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	}
	_, err := bedrockLLMAgainst(t, srv, bedrock.WithModel(converseClaude), bedrock.WithConverseAPI()).
		GenerateContent(t.Context(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeSystem, "rules"),
			llms.TextParts(llms.ChatMessageTypeHuman, "scan the host"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.TextPartWithReasoning("", thought),
				llms.ToolCall{ID: "call_1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "nmap", Arguments: `{}`}},
			}},
			llms.TextParts(llms.ChatMessageTypeAI, "checking"),
			{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{ToolCallID: "call_1", Name: "nmap", Content: "22/tcp open"}}},
			llms.TextParts(llms.ChatMessageTypeHuman, "go on"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("port 22 is open", thought)}},
			llms.TextParts(llms.ChatMessageTypeHuman, "report"),
		}, opts...)
	return err
}

func TestARefusedConverseHistoryNamesTheMessageInTheCallersChain(t *testing.T) {
	t.Parallel()

	const (
		bound = "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation. " +
			"Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\". The system prompt differs."
		modified = "messages.3.content.0: `thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified. " +
			"These blocks must remain as they were in the original response."
	)
	for _, streamed := range []bool{false, true} {
		operation, prefix := "Converse", "converse API call failed: "
		if streamed {
			operation, prefix = "ConverseStream", "converse stream API call failed: "
		}
		for _, tc := range []struct {
			message string
			fault   llms.HistoryFault
			chain   int
		}{
			{bound, llms.HistoryPrefixChanged, 2},
			{modified, llms.HistoryThinkingModified, 6},
		} {
			err := converseRefusal(t, tc.message, streamed)

			var rejected *llms.ErrHistoryRejected
			require.ErrorAs(t, err, &rejected, operation)
			require.Equal(t, tc.fault, rejected.Fault, operation)
			require.Equal(t, tc.chain, rejected.Message, operation)
			require.EqualError(t, err, prefix+"operation error Bedrock Runtime: "+operation+
				", https response error StatusCode: 400, RequestID: , ValidationException: "+tc.message)
			var apiErr smithy.APIError
			require.ErrorAs(t, err, &apiErr, operation)
			require.Equal(t, "ValidationException", apiErr.ErrorCode(), operation)
		}
	}
}

func TestARefusedBlockInAMergedConverseTurnNamesItsFirstMessage(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Amzn-Errortype", "ValidationException")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"message":"messages.1.content.0: Invalid `+"`signature`"+` in `+"`thinking`"+` block"}`)
	}))
	t.Cleanup(srv.Close)

	thought := (&reasoning.ContentReasoning{Content: "plan", Signature: []byte("sig")}).WrittenBy(converseClaude)
	_, err := bedrockLLMAgainst(t, srv, bedrock.WithModel(converseClaude), bedrock.WithConverseAPI()).
		GenerateContent(t.Context(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "a"),
			llms.TextParts(llms.ChatMessageTypeAI, "first"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.TextPartWithReasoning("", thought),
				llms.ToolCall{ID: "c1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "nmap", Arguments: `{}`}},
			}},
			{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{ToolCallID: "c1", Name: "nmap", Content: "open"}}},
			llms.TextParts(llms.ChatMessageTypeHuman, "b"),
		})

	var rejected *llms.ErrHistoryRejected
	require.ErrorAs(t, err, &rejected)
	require.Equal(t, 1, rejected.Message, "the thinking of message 2 rides in the turn Converse built from messages 1 and 2")
}

func TestAConversePromptOverTheContextWindowIsAnOverflow(t *testing.T) {
	t.Parallel()

	err := converseRefusal(t, "prompt is too long: 215000 tokens > 200000 maximum", false)

	var overflow *llms.ErrContextOverflow
	require.ErrorAs(t, err, &overflow)
	var apiErr smithy.APIError
	require.ErrorAs(t, err, &apiErr)
	require.Equal(t, "ValidationException", apiErr.ErrorCode())
	require.EqualError(t, err, "converse API call failed: operation error Bedrock Runtime: Converse, "+
		"https response error StatusCode: 400, RequestID: , ValidationException: prompt is too long: 215000 tokens > 200000 maximum")
}
