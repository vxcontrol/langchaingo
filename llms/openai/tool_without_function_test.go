package openai

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAToolWithoutAFunctionIsDroppedAndReported(t *testing.T) {
	t.Parallel()

	named := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "search", Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}
	body, warnings := hostCall(t, "http://api.openai.com/v1", "gpt-4.1",
		llms.WithTools([]llms.Tool{{Type: "function"}, {}, named}))
	tools, _ := body["tools"].([]any)
	require.Len(t, tools, 1, "%v", body)
	require.Equal(t, llms.WarningDrop, warnings["WithTools"].Kind, "%v", warnings)
	require.Equal(t, "2 tools", warnings["WithTools"].Asked)
}

func TestAHistoryCallOrToolTheDoorCannotSendIsRefusedBeforeTheRequest(t *testing.T) {
	t.Parallel()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL("http://api.openai.com/v1"), WithModel("gpt-4.1"), WithHTTPClient(doer))
	for name, call := range map[string]struct {
		messages []llms.MessageContent
		opts     []llms.CallOption
	}{
		"a history call without a function": {[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{ID: "call_1", Type: "function"}}},
			{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
				llms.ToolCallResponse{ToolCallID: "call_1", Name: "search", Content: "found"},
			}},
		}, nil},
		"a built-in tool the door cannot send": {
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			[]llms.CallOption{llms.WithTools([]llms.Tool{{Type: "web_search"}})},
		},
	} {
		_, err := llm.GenerateContent(context.Background(), call.messages, call.opts...)
		require.ErrorIs(t, err, llms.ErrInvalidRequest, name)
		require.Empty(t, doer.body, "%s: no request may go out", name)
	}
}

type cannedDoer string

func (d cannedDoer) Do(*http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(string(d))),
	}, nil
}

func TestALegacyFunctionCallFinishWithoutTheCallIsReadAsNoCall(t *testing.T) {
	t.Parallel()

	llm := newUnitLLM(t, WithBaseURL("http://api.openai.com/v1"), WithModel("gpt-4.1"),
		WithHTTPClient(cannedDoer(`{"id":"x","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"function_call"}]}`)))
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.NoError(t, err)
	require.Nil(t, resp.Choices[0].FuncCall)
}
