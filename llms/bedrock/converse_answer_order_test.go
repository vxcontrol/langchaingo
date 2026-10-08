package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func TestConverseSendsAnAnswerBackInTheOrderItCame(t *testing.T) {
	t.Parallel()

	var bodies [][]byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"output":{"message":{"role":"assistant","content":[`+
			`{"reasoningContent":{"reasoningText":{"text":"plan","signature":"sig1"}}},{"text":"A"},`+
			`{"toolUse":{"toolUseId":"tu1","name":"lookup","input":{"q":1}}},{"text":"B"}]}},`+
			`"stopReason":"tool_use","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv,
		bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}})
	question := llms.TextParts(llms.ChatMessageTypeHuman, "hi")

	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{question}, tools)
	require.NoError(t, err)
	require.Equal(t, "AB", resp.Choices[0].Content)

	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
		question,
		resp.Choices[0].Message(),
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "tu1", Name: "lookup", Content: "ok"},
		}},
	}, tools)
	require.NoError(t, err)

	var sent struct {
		Messages []struct {
			Content []any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(bodies[1], &sent))
	require.Equal(t, []any{
		map[string]any{"reasoningContent": map[string]any{"reasoningText": map[string]any{"text": "plan", "signature": "sig1"}}},
		map[string]any{"text": "A"},
		map[string]any{"toolUse": map[string]any{"toolUseId": "tu1", "name": "lookup", "input": map[string]any{"q": float64(1)}}},
		map[string]any{"text": "B"},
	}, sent.Messages[1].Content)
}

func TestAStreamedConverseAnswerKeepsItsOrder(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		enc := eventstream.NewEncoder()
		for _, event := range [][2]string{
			{"messageStart", `{"role":"assistant"}`},
			{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"plan"}}}`},
			{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"sig1"}}}`},
			{"contentBlockStop", `{"contentBlockIndex":0}`},
			{"contentBlockDelta", `{"contentBlockIndex":1,"delta":{"text":"A"}}`},
			{"contentBlockStop", `{"contentBlockIndex":1}`},
			{"contentBlockStart", `{"contentBlockIndex":2,"start":{"toolUse":{"toolUseId":"tu1","name":"lookup"}}}`},
			{"contentBlockDelta", `{"contentBlockIndex":2,"delta":{"toolUse":{"input":"{\"q\":1}"}}}`},
			{"contentBlockStop", `{"contentBlockIndex":2}`},
			{"contentBlockDelta", `{"contentBlockIndex":3,"delta":{"text":"B"}}`},
			{"contentBlockStop", `{"contentBlockIndex":3}`},
			{"messageStop", `{"stopReason":"tool_use"}`},
			{"metadata", `{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`},
		} {
			writeConverseEvent(t, w, enc, event[0], event[1])
		}
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv,
		bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(_ context.Context, _ streaming.Chunk) error { return nil }))
	require.NoError(t, err)
	require.Equal(t, "AB", resp.Choices[0].Content)

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

func TestAToolCallFinishedOnlyByTheStreamEndKeepsItsPlace(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		enc := eventstream.NewEncoder()
		for _, event := range [][2]string{
			{"messageStart", `{"role":"assistant"}`},
			{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"A"}}`},
			{"contentBlockStart", `{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"tu1","name":"lookup"}}}`},
			{"contentBlockDelta", `{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{}"}}}`},
			{"contentBlockDelta", `{"contentBlockIndex":2,"delta":{"text":"B"}}`},
			{"contentBlockStop", `{"contentBlockIndex":2}`},
			{"messageStop", `{"stopReason":"tool_use"}`},
			{"metadata", `{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`},
		} {
			writeConverseEvent(t, w, enc, event[0], event[1])
		}
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv,
		bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(_ context.Context, _ streaming.Chunk) error { return nil }))
	require.NoError(t, err)

	parts := resp.Choices[0].Parts
	require.Len(t, parts, 3)
	require.Equal(t, llms.TextContent{Text: "A"}, parts[0])
	call, ok := parts[1].(llms.ToolCall)
	require.True(t, ok, "%#v", parts[1])
	require.Equal(t, "tu1", call.ID)
	require.Equal(t, llms.TextContent{Text: "B"}, parts[2])
}
