package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func TestProbeLogProbs(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop",
			"logprobs":{"content":[{"token":"ok","logprob":-0.1,"bytes":[111,107],"top_logprobs":[{"token":"ok","logprob":-0.1,"bytes":[111,107]}]}]}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	llm, err := openai.New(openai.WithBaseURL(srv.URL), openai.WithToken("x"), openai.WithModel("gpt-4o"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithLogProbs(true), llms.WithTopLogProbs(1))
	t.Logf("err=%v req=%s", err, body)
	c := resp.Choices[0]
	keys := []string{}
	for k := range c.GenerationInfo {
		keys = append(keys, k)
	}
	b, _ := json.Marshal(c)
	t.Logf("choice=%s", b)
	resp, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithTopLogProbs(3))
	t.Logf("toplogprobs only: err=%v req=%s", err, body)
}
