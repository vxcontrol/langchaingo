package anthropic_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func v3PartialRun(t *testing.T, label, tail string, failConsumer bool) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n")
		fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"delete_files\",\"input\":{}}}\n\n")
		fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\": \\\"/tmp/bui\"}}\n\n")
		fmt.Fprint(w, tail)
	}))
	defer srv.Close()
	llm, err := anthropic.New(anthropic.WithToken("k"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-sonnet-4-5"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "clean up")},
		llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{Name: "delete_files", Parameters: map[string]any{"type": "object"}}}}),
		llms.WithStreamingFunc(func(_ context.Context, c streaming.Chunk) error {
			if failConsumer && c.Type == streaming.ChunkTypeToolCall {
				return errors.New("consumer gave up")
			}
			return nil
		}))
	if resp == nil {
		fmt.Printf("[%s] err=%v resp=nil\n", label, err)
		return
	}
	for _, tc := range resp.Choices[0].ToolCalls {
		fmt.Printf("[%s] err=%v stop=%q toolcall id=%s name=%s args=%q\n", label, err != nil, resp.Choices[0].StopReason, tc.ID, tc.FunctionCall.Name, tc.FunctionCall.Arguments)
	}
	if len(resp.Choices[0].ToolCalls) == 0 {
		fmt.Printf("[%s] resp without toolcalls\n", label)
	}
}

func TestProbeV3Partial(t *testing.T) {
	v3PartialRun(t, "error-event", "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n", false)
	v3PartialRun(t, "malformed", "data: {not json\n\n", false)
	v3PartialRun(t, "consumer-err", "", true)
}
