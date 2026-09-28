package openai_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

type vfDoer struct {
	body []byte
	n    int
}

func (d *vfDoer) Do(req *http.Request) (*http.Response, error) {
	d.body, _ = io.ReadAll(req.Body)
	d.n++
	resp := `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	return &http.Response{StatusCode: 200, Header: h, Body: io.NopCloser(bytes.NewReader([]byte(resp))), Request: req}, nil
}

func TestProbeVerF12(t *testing.T) {
	or := "https://openrouter.ai/api/v1"
	ll := "http://litellm.local:4000/v1"
	type sc struct {
		name, base, model string
		modern            bool
		opt               llms.CallOption
	}
	ad := llms.WithAdaptiveReasoning("")
	none := llms.WithReasoning("none", 0)
	for _, s := range []sc{
		{"F1 or-modern-opus46", or, "anthropic/claude-opus-4.6", true, ad},
		{"F1 or-legacy-opus46", or, "anthropic/claude-opus-4.6", false, ad},
		{"F1 litellm-haiku45", ll, "claude-haiku-4-5", false, ad},
		{"F1 litellm-sonnet45", ll, "claude-sonnet-4-5", false, ad},
		{"F1 litellm-sonnet37", ll, "claude-3-7-sonnet-latest", false, ad},
		{"F1 litellm-opus46", ll, "claude-opus-4-6", false, ad},
		{"F1 or-legacy-gpt51(optin)", or, "openai/gpt-5.1", false, ad},
		{"F2 openai-gpt51-none", "https://api.openai.com/v1", "gpt-5.1", false, none},
		{"F2 openai-gpt52-none", "https://api.openai.com/v1", "gpt-5.2", false, none},
		{"F2 or-modern-gpt51-none", or, "openai/gpt-5.1", true, none},
	} {
		d := &vfDoer{}
		o := []openai.Option{openai.WithToken("k"), openai.WithBaseURL(s.base), openai.WithHTTPClient(d), openai.WithModel(s.model)}
		if s.modern {
			o = append(o, openai.WithModernReasoningFormat())
		}
		llm, err := openai.New(o...)
		if err != nil {
			fmt.Println(s.name, "NEW", err)
			continue
		}
		resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hello")}, s.opt)
		fmt.Printf("[%s] sent=%d REQ=%s ERR=%v\n", s.name, d.n, d.body, err)
		printWarnVF(s.name, resp)
	}
}
