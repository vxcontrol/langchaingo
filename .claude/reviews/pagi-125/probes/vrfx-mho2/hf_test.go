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

func vrfxHF(t *testing.T, reply string) (*LLM, *[]byte) {
	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(srv.Close)
	llm, err := New(WithToken("t"), WithURL(srv.URL), WithModel("Qwen/Qwen3-32B"))
	if err != nil {
		t.Fatal(err)
	}
	return llm, &raw
}

func TestProbeVrfxHFSO(t *testing.T) {
	llm, _ := vrfxHF(t, `{"choices":[{"message":{"content":"Sure! Paris is the capital."},"finish_reason":"stop"}]}`)
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "capital of France")},
		llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "capital", Schema: json.RawMessage(`{"type":"object","properties":{"capital":{"type":"string"}},"required":["capital"],"additionalProperties":false}`)}))
	fmt.Printf("HF SO: err=%v content=%q\n", err, resp.Choices[0].Content)
	for _, w := range resp.Warnings {
		fmt.Println("  warning:", w.String())
	}
}

func TestProbeVrfxHFBudget(t *testing.T) {
	llm, raw := vrfxHF(t, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`)
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithMaxTokens(4096), llms.WithReasoning("", 800))
	var body map[string]any
	_ = json.Unmarshal(*raw, &body)
	fmt.Printf("HF budget: err=%v wire reasoning_effort=%#v\n", err, body["reasoning_effort"])
	for _, w := range resp.Warnings {
		fmt.Printf("  warning: %s | kind=%s sent=%q\n", w.String(), w.Kind, w.Sent)
	}
}
