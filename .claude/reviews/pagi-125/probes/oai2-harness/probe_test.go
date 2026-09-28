package openai_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

type capDoer struct {
	body   []byte
	status int
	resp   string
	sse    bool
}

func (d *capDoer) Do(req *http.Request) (*http.Response, error) {
	b, _ := io.ReadAll(req.Body)
	d.body = b
	status := d.status
	if status == 0 {
		status = 200
	}
	resp := d.resp
	if resp == "" {
		resp = `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	}
	h := http.Header{}
	if d.sse {
		h.Set("Content-Type", "text/event-stream")
	} else {
		h.Set("Content-Type", "application/json")
	}
	return &http.Response{StatusCode: status, Header: h, Body: io.NopCloser(bytes.NewReader([]byte(resp))), Request: req}, nil
}

type scen struct {
	name    string
	base    string
	model   string
	cliOpts []openai.Option
	opts    []llms.CallOption
	msgs    []llms.MessageContent
	resp    string
	sse     bool
	status  int
}

func run(t *testing.T, s scen) {
	d := &capDoer{resp: s.resp, sse: s.sse, status: s.status}
	base := s.base
	if base == "" {
		base = "http://127.0.0.1:1/v1"
	}
	copts := []openai.Option{openai.WithToken("k"), openai.WithBaseURL(base), openai.WithHTTPClient(d), openai.WithModel(s.model)}
	copts = append(copts, s.cliOpts...)
	llm, err := openai.New(copts...)
	if err != nil {
		fmt.Printf("[%s] NEW ERR: %v\n", s.name, err)
		return
	}
	msgs := s.msgs
	if msgs == nil {
		msgs = []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hello")}
	}
	resp, err := llm.GenerateContent(context.Background(), msgs, s.opts...)
	body := string(d.body)
	fmt.Printf("[%s] REQ: %s\n", s.name, body)
	if err != nil {
		fmt.Printf("[%s] ERR: %v\n", s.name, err)
	}
	if resp != nil {
		for i, c := range resp.Choices {
			fmt.Printf("[%s] CHOICE %d: content=%q stop=%q reasoning=%v tools=%d\n", s.name, i, c.Content, c.StopReason, c.Reasoning != nil, len(c.ToolCalls))
		}
	}
	printWarn(s.name, resp)
	_ = strings.TrimSpace
	_ = os.Getenv
}

func TestProbeHarness(t *testing.T) {
	for _, s := range scenarios() {
		run(t, s)
	}
}
