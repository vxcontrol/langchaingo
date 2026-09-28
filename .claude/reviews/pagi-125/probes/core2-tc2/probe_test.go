package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func TestProbeToolChoiceWire(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"gpt-4.1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	llm, err := openai.New(openai.WithToken("k"), openai.WithBaseURL(srv.URL), openai.WithModel("gpt-4.1"))
	if err != nil {
		t.Fatal(err)
	}
	tools := []llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{Name: "calc", Parameters: map[string]any{"type": "object"}}}, {Type: "function", Function: &llms.FunctionDefinition{Name: "search", Parameters: map[string]any{"type": "object"}}}}
	choice := map[string]any{"type": "function", "function": map[string]string{"name": "calc"}}
	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithTools(tools), llms.WithToolChoice(choice))
	out, _ := json.Marshal(body["tool_choice"])
	fmt.Printf("PROBE err=%v tool_choice on wire=%s\n", err, out)
}
