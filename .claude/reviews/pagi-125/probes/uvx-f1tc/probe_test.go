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

func TestProbeUvxTC(t *testing.T) {
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		tc, _ := json.Marshal(m["tool_choice"])
		got = append(got, string(tc))
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"gpt-4.1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	llm, err := openai.New(openai.WithBaseURL(srv.URL), openai.WithToken("k"), openai.WithModel("gpt-4.1"))
	if err != nil {
		t.Fatal(err)
	}
	tools := []llms.Tool{
		{Type: "function", Function: &llms.FunctionDefinition{Name: "calc", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: &llms.FunctionDefinition{Name: "search", Parameters: map[string]any{"type": "object"}}},
	}
	choices := map[string]any{
		"map-any":      map[string]any{"type": "function", "function": map[string]any{"name": "calc"}},
		"map-string":   map[string]any{"type": "function", "function": map[string]string{"name": "calc"}},
		"map-funcref":  map[string]any{"type": "function", "function": llms.FunctionReference{Name: "calc"}},
		"map-*funcref": map[string]any{"type": "function", "function": &llms.FunctionReference{Name: "calc"}},
		"struct":       llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "calc"}},
	}
	for _, k := range []string{"map-any", "map-string", "map-funcref", "map-*funcref", "struct"} {
		got = nil
		_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithTools(tools), llms.WithToolChoice(choices[k]))
		fmt.Printf("%-13s err=%v wire=%v\n", k, err, got)
	}
}
