package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func extraBodyWire(t *testing.T, model string, client []Option, answer string, opts ...llms.CallOption) map[string]json.RawMessage {
	t.Helper()

	content, err := json.Marshal(answer)
	if err != nil {
		t.Fatalf("answer: %v", err)
	}
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		if strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m",`+
				`"choices":[{"index":0,"delta":{"role":"assistant","content":`+string(content)+`},"finish_reason":"stop"}]}`+
				"\n\ndata: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":`+string(content)+`},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(append([]Option{WithBaseURL(srv.URL), WithToken("test"), WithModel(model)}, client...)...)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if _, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...); err != nil {
		t.Fatalf("GenerateContent() error: %v", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatalf("the wire is not a JSON object: %v, body: %s", err, body)
	}
	return fields
}

func TestExtraBodyLeavesTheFieldsItDoesNotNameAsTheDoorBuiltThem(t *testing.T) {
	t.Parallel()

	seed := llms.WithSeed(9007199254740993)
	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{
		Name: "verdict",
		Schema: json.RawMessage(`{"type":"object","properties":{"reasoning":{"type":"string"},` +
			`"answer":{"type":"string"}},"required":["reasoning","answer"],"additionalProperties":false}`),
	})
	const answer = `{"reasoning":"r","answer":"a"}`
	without := extraBodyWire(t, "gpt-4o", nil, answer, seed, schema)
	with := extraBodyWire(t, "gpt-4o", nil, answer,
		seed, schema, llms.WithExtraBody(map[string]any{"preserve_thinking": true}))

	for _, field := range []string{"seed", "response_format", "messages"} {
		if string(with[field]) != string(without[field]) {
			t.Errorf("%s changed under extra body:\nwant %s\ngot  %s", field, without[field], with[field])
		}
	}
	if string(with["preserve_thinking"]) != "true" {
		t.Errorf("the extra body field did not reach the wire: %s", with["preserve_thinking"])
	}
}
