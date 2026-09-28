package anthropic_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func TestProbeVfyForced(t *testing.T) {
	const reply = `{"id":"x","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{Name: "calc", Description: "d",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}
	for _, model := range []string{"claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5", "claude-sonnet-4-5"} {
		for _, rs := range []bool{false, true} {
			var sent map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(b, &sent)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, reply)
			}))
			llm, _ := anthropic.New(anthropic.WithToken("k"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel(model))
			opts := []llms.CallOption{llms.WithTools([]llms.Tool{tool}), llms.WithToolChoice("any"), llms.WithMaxTokens(8000)}
			if rs {
				opts = append(opts, llms.WithReasoning(llms.ReasoningHigh, 0))
			}
			_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
			fmt.Printf("%-18s reasoning=%-5v err=%v sent_tool_choice=%v thinking=%v\n", model, rs, err, sent["tool_choice"], sent["thinking"])
			srv.Close()
		}
	}
}

func TestProbeVfyCleanEOF(t *testing.T) {
	body := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\",\"content\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"sixty rooms are free\"}}\n\n"
	for _, tail := range []string{"", "data: {not json\n\n"} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, body+tail)
		}))
		llm, _ := anthropic.New(anthropic.WithToken("k"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-sonnet-4-5"))
		var streamed string
		resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithStreamingFunc(func(_ context.Context, c streaming.Chunk) error {
				if c.Type == streaming.ChunkTypeText {
					streamed += c.Content
				}
				return nil
			}))
		got := "<nil>"
		if resp != nil && len(resp.Choices) > 0 {
			got = fmt.Sprintf("%q", resp.Choices[0].Content)
		}
		fmt.Printf("malformedTail=%v streamed=%q err=%v resp=%s\n", tail != "", streamed, err, got)
		srv.Close()
	}
}
