package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func mho2HF(t *testing.T, reply string) (*LLM, *[]byte, *string) {
	t.Helper()
	var raw []byte
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(srv.Close)
	llm, err := New(WithToken("t"), WithURL(srv.URL), WithModel("Qwen/Qwen3-32B"))
	if err != nil {
		t.Fatal(err)
	}
	return llm, &raw, &path
}

func TestProbeHFBudgetWarning(t *testing.T) {
	llm, raw, _ := mho2HF(t, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`)
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithMaxTokens(4096), llms.WithReasoning("", 800))
	fmt.Println("err:", err)
	var body map[string]any
	_ = json.Unmarshal(*raw, &body)
	fmt.Printf("wire reasoning_effort: %#v\n", body["reasoning_effort"])
	for _, w := range resp.Warnings {
		fmt.Printf("warning: %s\n", w.String())
	}
}

func TestProbeHFStructuredOutput(t *testing.T) {
	llm, _, _ := mho2HF(t, `{"choices":[{"message":{"content":"Sure! Paris is the capital."},"finish_reason":"stop"}]}`)
	schema := json.RawMessage(`{"type":"object","properties":{"capital":{"type":"string"}},"required":["capital"],"additionalProperties":false}`)
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "capital of France")},
		llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "capital", Schema: schema}))
	fmt.Printf("HF structured output: err=%v content=%q\n", err, resp.Choices[0].Content)
	for _, w := range resp.Warnings {
		fmt.Printf("warning: %s\n", w.String())
	}
}

func TestProbeHFSystemFirst(t *testing.T) {
	llm, raw, _ := mho2HF(t, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "You are terse."),
		llms.TextParts(llms.ChatMessageTypeHuman, "What is 2+2?"),
	})
	fmt.Println("err:", err, "body:", string(*raw))
}
