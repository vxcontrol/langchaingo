package openai

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAReplayedCallWithoutArgumentsSendsAnEmptyObject(t *testing.T) {
	t.Parallel()

	for _, arguments := range []string{"null", "", "  "} {
		t.Run(fmt.Sprintf("%q", arguments), func(t *testing.T) {
			t.Parallel()

			var raw []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"m",`+
					`"choices":[{"index":0,"message":{"role":"assistant","content":"noon"},"finish_reason":"stop"}],`+
					`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			}))
			t.Cleanup(srv.Close)

			llm, err := New(WithBaseURL(srv.URL), WithToken("token"), WithModel("qwen3.6-plus"))
			require.NoError(t, err)

			_, err = llm.GenerateContent(t.Context(), []llms.MessageContent{
				llms.TextParts(llms.ChatMessageTypeHuman, "what time is it?"),
				{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{
					ID: "c1", Type: "function",
					FunctionCall: &llms.FunctionCall{Name: "clock", Arguments: arguments},
				}}},
				{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{
					ToolCallID: "c1", Name: "clock", Content: "noon",
				}}},
			})
			require.NoError(t, err)

			var sent struct {
				Messages []struct {
					ToolCalls []struct {
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(raw, &sent))
			require.Len(t, sent.Messages, 3)
			require.Len(t, sent.Messages[1].ToolCalls, 1)
			assert.Equal(t, "{}", sent.Messages[1].ToolCalls[0].Function.Arguments)
		})
	}
}
