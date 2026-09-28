package openai_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func TestProbeSkv(t *testing.T) {
	var last string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		last = string(b)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	cases := []struct {
		name   string
		model  string
		modern bool
		opt    llms.CallOption
	}{
		{"modern-opus46-adaptive", "anthropic/claude-opus-4.6", true, llms.WithAdaptiveReasoning("")},
		{"legacy-haiku45-adaptive", "anthropic/claude-haiku-4-5", false, llms.WithAdaptiveReasoning("")},
		{"legacy-sonnet45-adaptive", "anthropic/claude-sonnet-4-5", false, llms.WithAdaptiveReasoning("")},
		{"gpt51-none", "gpt-5.1", false, llms.WithReasoning("none", 0)},
		{"gpt51-none-modern", "openai/gpt-5.1", true, llms.WithReasoning("none", 0)},
	}
	for _, c := range cases {
		last = ""
		opts := []openai.Option{openai.WithBaseURL(srv.URL), openai.WithToken("x"), openai.WithModel(c.model)}
		if c.modern {
			opts = append(opts, openai.WithModernReasoningFormat())
		}
		llm, err := openai.New(opts...)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, c.opt)
		fmt.Printf("[%s] ERR=%v\nREQ=%s\n", c.name, err, last)
		if resp != nil {
			v := reflect.ValueOf(resp).Elem().FieldByName("Warnings")
			if v.IsValid() {
				fmt.Printf("WARN=%+v\n", v.Interface())
			}
		}
	}
}
