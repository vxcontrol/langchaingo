package bedrock_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func streamedCallOf(t *testing.T, llm *bedrock.LLM) (llms.ToolCall, []streaming.ToolCall) {
	t.Helper()

	var streamed []streaming.ToolCall
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time is it")},
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			if chunk.Type == streaming.ChunkTypeToolCall {
				streamed = append(streamed, chunk.ToolCall)
			}
			return nil
		}))
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	require.Len(t, resp.Choices[0].ToolCalls, 1)
	require.NotNil(t, resp.Choices[0].ToolCalls[0].FunctionCall)

	return resp.Choices[0].ToolCalls[0], streamed
}

func TestAConverseStreamedCallWithoutArgumentsReachesTheCallerAsAnEmptyObject(t *testing.T) {
	t.Parallel()

	for name, deltas := range map[string][]string{
		"no input delta":    nil,
		"empty input delta": {`{"contentBlockIndex":0,"delta":{"toolUse":{"input":""}}}`},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
				enc := eventstream.NewEncoder()
				writeConverseEvent(t, w, enc, "messageStart", `{"role":"assistant"}`)
				writeConverseEvent(t, w, enc, "contentBlockStart",
					`{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"t1","name":"get_time"}}}`)
				for _, delta := range deltas {
					writeConverseEvent(t, w, enc, "contentBlockDelta", delta)
				}
				writeConverseEvent(t, w, enc, "contentBlockStop", `{"contentBlockIndex":0}`)
				writeConverseEvent(t, w, enc, "messageStop", `{"stopReason":"tool_use"}`)
				writeConverseEvent(t, w, enc, "metadata",
					`{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
			}))
			t.Cleanup(srv.Close)

			call, streamed := streamedCallOf(t, bedrockLLMAgainst(t, srv,
				bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI()))

			assert.JSONEq(t, `{}`, call.FunctionCall.Arguments)
			require.Len(t, streamed, 1)
			assert.JSONEq(t, `{}`, streamed[0].Arguments)
		})
	}
}

func TestALegacyStreamedClaudeCallWithoutArgumentsReachesTheCallerAsAnEmptyObject(t *testing.T) {
	t.Parallel()

	for name, deltas := range map[string][]string{
		"no input delta": nil,
		"empty input delta": {`{"type":"content_block_delta","index":0,` +
			`"delta":{"type":"input_json_delta","partial_json":""}}`},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
				enc := eventstream.NewEncoder()
				writeLegacyChunk(t, w, enc, `{"type":"message_start","message":{"id":"x","type":"message",`+
					`"role":"assistant","model":"m","content":[],"stop_reason":null,`+
					`"usage":{"input_tokens":1,"output_tokens":1}}}`)
				writeLegacyChunk(t, w, enc, `{"type":"content_block_start","index":0,`+
					`"content_block":{"type":"tool_use","id":"t1","name":"get_time","input":{}}}`)
				for _, delta := range deltas {
					writeLegacyChunk(t, w, enc, delta)
				}
				writeLegacyChunk(t, w, enc, `{"type":"content_block_stop","index":0}`)
				writeLegacyChunk(t, w, enc, `{"type":"message_delta","delta":{"stop_reason":"tool_use"},`+
					`"usage":{"output_tokens":1}}`)
			}))
			t.Cleanup(srv.Close)

			call, _ := streamedCallOf(t, bedrockLLMAgainst(t, srv,
				bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0")))

			assert.JSONEq(t, `{}`, call.FunctionCall.Arguments)
		})
	}
}
