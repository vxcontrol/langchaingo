package anthropic_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func probeServe(t *testing.T, body string) *anthropic.LLM {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	llm, err := anthropic.New(anthropic.WithToken("k"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-haiku-4-5"))
	if err != nil {
		t.Fatal(err)
	}
	return llm
}

const probeHead = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"delete_files","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\": \"/tmp/bui"}}

`

func TestProbePartialToolCall(t *testing.T) {
	for name, tail := range map[string]string{
		"error event": "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"overloaded\"}}\n",
		"malformed":   "data: {not json\n",
	} {
		llm := probeServe(t, probeHead+tail)
		resp, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "clean up")},
			llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
		fmt.Printf("[%s] err=%v\n", name, err)
		if resp != nil && len(resp.Choices) > 0 {
			for _, tc := range resp.Choices[0].ToolCalls {
				fmt.Printf("[%s] partial tool call id=%s name=%s args=%q\n", name, tc.ID, tc.FunctionCall.Name, tc.FunctionCall.Arguments)
			}
			fmt.Printf("[%s] stop=%q\n", name, resp.Choices[0].StopReason)
		} else {
			fmt.Printf("[%s] resp=nil\n", name)
		}
	}
}

func TestProbeCleanEOF(t *testing.T) {
	body := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"sixty rooms are free"}}

`
	llm := probeServe(t, body)
	var got string
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "rooms?")},
		llms.WithStreamingFunc(func(_ context.Context, c streaming.Chunk) error { got += c.Content; return nil }))
	fmt.Printf("[clean EOF] streamed=%q err=%v resp=%v\n", got, err, resp)
}
