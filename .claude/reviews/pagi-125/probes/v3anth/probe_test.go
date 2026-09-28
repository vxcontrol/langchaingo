package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const v3Stream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","model":"claude-opus-4-8","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":1,"service_tier":"standard","speed":"fast"}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}
`

const v3JSON = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3,"service_tier":"standard","speed":"fast"}}`

func v3LLM(t *testing.T, model string, bodies *[]string) *anthropic.LLM {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		*bodies = append(*bodies, string(b))
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		if s, _ := m["stream"].(bool); s {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, v3Stream)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, v3JSON)
	}))
	t.Cleanup(srv.Close)
	llm, err := anthropic.New(anthropic.WithToken("k"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel(model))
	if err != nil {
		t.Fatal(err)
	}
	return llm
}

func TestProbeV3Speed(t *testing.T) {
	var bodies []string
	llm := v3LLM(t, "claude-opus-4-8", &bodies)
	ask := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
	r, err := llm.GenerateContent(context.Background(), ask, llms.WithMaxTokens(100), llms.WithInferenceSpeed("fast"))
	if err == nil {
		t.Logf("nonstream: speed=%q tier=%q", r.Choices[0].GenerationInfo["InferenceSpeed"], r.Choices[0].GenerationInfo["ServiceTier"])
	} else {
		t.Logf("nonstream err=%v", err)
	}
	r, err = llm.GenerateContent(context.Background(), ask, llms.WithMaxTokens(100), llms.WithInferenceSpeed("fast"),
		llms.WithStreamingFunc(func(_ context.Context, _ streaming.Chunk) error { return nil }))
	if err == nil {
		t.Logf("stream: content=%q speed=%q tier=%q", r.Choices[0].Content, r.Choices[0].GenerationInfo["InferenceSpeed"], r.Choices[0].GenerationInfo["ServiceTier"])
	} else {
		t.Logf("stream err=%v", err)
	}
	for _, b := range bodies {
		var m map[string]any
		_ = json.Unmarshal([]byte(b), &m)
		t.Logf("wire speed=%v stream=%v", m["speed"], m["stream"])
	}
}

func TestProbeV3Off(t *testing.T) {
	for _, model := range []string{"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"} {
		var bodies []string
		llm := v3LLM(t, model, &bodies)
		ask := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
		_, err := llm.GenerateContent(context.Background(), ask, llms.WithMaxTokens(100), llms.WithReasoningDisabled())
		th := "<no request>"
		if len(bodies) > 0 {
			var m map[string]json.RawMessage
			_ = json.Unmarshal([]byte(bodies[0]), &m)
			th = string(m["thinking"])
		}
		t.Logf("%s off: thinking=%s err=%v", model, th, err)
	}
}
