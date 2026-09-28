package anthropic_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const bigToolArgument = "10000000000000001"

func TestAnAnsweredToolCallKeepsEveryDigitTheVendorSent(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_test","type":"message","role":"assistant",`+
			`"model":"claude-opus-4-6","content":[`+
			`{"type":"tool_use","id":"toolu_1","name":"charge",`+
			`"input":{"account":`+bigToolArgument+`}}],`+
			`"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-opus-4-6"))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "charge that account")},
		llms.WithMaxTokens(64))
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	require.Len(t, resp.Choices[0].ToolCalls, 1)

	assert.Contains(t, resp.Choices[0].ToolCalls[0].FunctionCall.Arguments, bigToolArgument)
}

func TestAReplayedToolCallCarriesTheSameDigitsBack(t *testing.T) {
	t.Parallel()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_test","type":"message","role":"assistant",`+
			`"model":"claude-opus-4-6","content":[{"type":"text","text":"done"}],`+
			`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-opus-4-6"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "charge that account"),
			{
				Role: llms.ChatMessageTypeAI,
				Parts: []llms.ContentPart{llms.ToolCall{
					ID:   "toolu_1",
					Type: "function",
					FunctionCall: &llms.FunctionCall{
						Name:      "charge",
						Arguments: `{"account":` + bigToolArgument + `}`,
					},
				}},
			},
			llms.TextParts(llms.ChatMessageTypeHuman, "did it go through?"),
		}, llms.WithMaxTokens(64))
	require.NoError(t, err)

	assert.Contains(t, string(body), bigToolArgument)
}

func TestAStreamedToolCallKeepsTheDigitsItsDeltasSpelledOut(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":"+
			"{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-6\","+
			"\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n")
		_, _ = io.WriteString(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\","+
			"\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\","+
			"\"name\":\"charge\",\"input\":{}}}\n\n")
		_, _ = io.WriteString(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\","+
			"\"index\":0,\"delta\":{\"type\":\"input_json_delta\","+
			"\"partial_json\":\"{\\\"account\\\":"+bigToolArgument+"}\"}}\n\n")
		_, _ = io.WriteString(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
		_, _ = io.WriteString(w, "event: message_delta\ndata: {\"type\":\"message_delta\","+
			"\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}\n\n")
		_, _ = io.WriteString(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-opus-4-6"))
	require.NoError(t, err)

	var streamed []string
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "charge that account")},
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			if chunk.ToolCall.Arguments != "" {
				streamed = append(streamed, chunk.ToolCall.Arguments)
			}
			return nil
		}))
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	require.Len(t, resp.Choices[0].ToolCalls, 1)

	assert.Contains(t, resp.Choices[0].ToolCalls[0].FunctionCall.Arguments, bigToolArgument)
	require.NotEmpty(t, streamed, "the door must hand the caller the call it assembled")
	assert.Contains(t, streamed[len(streamed)-1], bigToolArgument)
}
