package anthropic_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/internal/toolcall"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
)

func TestAReplayedCallWithoutArgumentsSendsAnEmptyInputObject(t *testing.T) {
	t.Parallel()

	for _, arguments := range []string{"null", "{}", "", "  "} {
		t.Run(fmt.Sprintf("%q", arguments), func(t *testing.T) {
			t.Parallel()

			var body []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"msg_test","type":"message","role":"assistant",`+
					`"model":"claude-sonnet-4-5","content":[{"type":"text","text":"12:34"}],`+
					`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
			}))
			t.Cleanup(srv.Close)

			llm, err := anthropic.New(anthropic.WithToken("test-key"),
				anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-sonnet-4-5"))
			require.NoError(t, err)

			_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
				llms.TextParts(llms.ChatMessageTypeHuman, "what time is it"),
				{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{
					ID: "toolu_1", Type: "function",
					FunctionCall: &llms.FunctionCall{Name: "get_time", Arguments: arguments},
				}}},
				{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
					llms.ToolCallResponse{ToolCallID: "toolu_1", Name: "get_time", Content: "12:34"},
				}},
			}, llms.WithMaxTokens(64))
			require.NoError(t, err)

			var sent struct {
				Messages []struct {
					Content []map[string]json.RawMessage `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(body, &sent))
			require.Len(t, sent.Messages, 3)
			toolUse := sent.Messages[1].Content[0]
			require.JSONEq(t, `"tool_use"`, string(toolUse["type"]))
			assert.JSONEq(t, `{}`, string(toolUse["input"]),
				"the Messages API takes input as a required object, so a call without arguments carries {}")
		})
	}
}

func TestAReplayedCallWithDataAfterItsArgumentsIsRefusedBeforeTheNetwork(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-sonnet-4-5"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "book both rooms"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{
			ID: "toolu_1", Type: "function",
			FunctionCall: &llms.FunctionCall{Name: "book", Arguments: `{"room":1}{"room":2}`},
		}}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "toolu_1", Name: "book", Content: "done"},
		}},
	}, llms.WithMaxTokens(64))

	require.ErrorIs(t, err, toolcall.ErrNotAnObject)
	assert.Zero(t, requests.Load(), "the second object would be dropped, so nothing may reach the vendor")
}
