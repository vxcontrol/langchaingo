package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func TestAnAnswerGoesBackInTheOrderItCame(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name    string
		content string
		replay  []any
	}{
		{
			name: "text around a tool call",
			content: `[{"type":"thinking","thinking":"plan","signature":"sig1"},{"type":"text","text":"A"},` +
				`{"type":"tool_use","id":"tu1","name":"lookup","input":{"q":1}},{"type":"text","text":"B"}]`,
			replay: []any{
				map[string]any{"type": "thinking", "thinking": "plan", "signature": "sig1"},
				map[string]any{"type": "text", "text": "A"},
				map[string]any{"type": "tool_use", "id": "tu1", "name": "lookup", "input": map[string]any{"q": float64(1)}},
				map[string]any{"type": "text", "text": "B"},
			},
		},
		{
			name: "a tool call before any text",
			content: `[{"type":"thinking","thinking":"plan","signature":"sig1"},` +
				`{"type":"tool_use","id":"tu1","name":"lookup","input":{}},{"type":"text","text":"B"}]`,
			replay: []any{
				map[string]any{"type": "thinking", "thinking": "plan", "signature": "sig1"},
				map[string]any{"type": "tool_use", "id": "tu1", "name": "lookup", "input": map[string]any{}},
				map[string]any{"type": "text", "text": "B"},
			},
		},
		{
			name:    "two text blocks stay two",
			content: `[{"type":"text","text":"A"},{"type":"text","text":"B"}]`,
			replay: []any{
				map[string]any{"type": "text", "text": "A"},
				map[string]any{"type": "text", "text": "B"},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var bodies [][]byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				bodies = append(bodies, body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-5","content":`+
					tc.content+`,"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`)
			}))
			t.Cleanup(srv.Close)

			llm, err := anthropic.New(
				anthropic.WithToken("test-key"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-sonnet-5"))
			require.NoError(t, err)
			tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
				Name: "lookup", Parameters: map[string]any{"type": "object"},
			}}})
			question := llms.TextParts(llms.ChatMessageTypeHuman, "hi")

			resp, err := llm.GenerateContent(t.Context(), []llms.MessageContent{question}, tools)
			require.NoError(t, err)
			_, err = llm.GenerateContent(t.Context(), []llms.MessageContent{
				question,
				resp.Choices[0].Message(),
				llms.TextParts(llms.ChatMessageTypeHuman, "go on"),
			}, tools)
			require.NoError(t, err)

			var sent struct {
				Messages []struct {
					Content []any `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(bodies[1], &sent))
			require.Equal(t, tc.replay, sent.Messages[1].Content)
		})
	}
}

func TestTheAnswerTextStaysJoinedForExistingCallers(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-5",`+
			`"content":[{"type":"text","text":"A"},{"type":"text","text":"B"}],"stop_reason":"end_turn",`+
			`"usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(
		anthropic.WithToken("test-key"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-sonnet-5"))
	require.NoError(t, err)
	resp, err := llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.NoError(t, err)
	require.Equal(t, "AB", resp.Choices[0].Content)
}

func TestAStreamedAnswerKeepsItsOrder(t *testing.T) {
	t.Parallel()

	llm := answeringLLM(t, "text/event-stream", `event: message_start
data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig1"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"A"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu1","name":"lookup","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\":1}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: content_block_start
data: {"type":"content_block_start","index":3,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"B"}}

event: content_block_stop
data: {"type":"content_block_stop","index":3}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
`)

	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(_ context.Context, _ streaming.Chunk) error { return nil }))
	require.NoError(t, err)

	parts := resp.Choices[0].Parts
	require.Len(t, parts, 3)
	first, ok := parts[0].(llms.TextContent)
	require.True(t, ok)
	require.Equal(t, "A", first.Text)
	require.NotEmpty(t, first.Reasoning.Sequence())
	require.Equal(t, "plan", first.Reasoning.Sequence()[0].Text)
	call, ok := parts[1].(llms.ToolCall)
	require.True(t, ok)
	require.Equal(t, "tu1", call.ID)
	require.Equal(t, llms.TextContent{Text: "B"}, parts[2])
}
