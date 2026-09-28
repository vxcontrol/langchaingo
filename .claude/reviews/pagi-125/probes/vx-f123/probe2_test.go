package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func TestProbeVXF2(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"gpt-4o",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop",
			"logprobs":{"content":[{"token":"ok","logprob":-0.01,"bytes":[111,107],"top_logprobs":[{"token":"ok","logprob":-0.01,"bytes":[111,107]}]}]}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	target, _ := url.Parse(srv.URL)
	llm, err := openai.New(openai.WithBaseURL("https://api.openai.com/v1"), openai.WithToken("x"), openai.WithModel("gpt-4o"), openai.WithHTTPClient(&http.Client{Transport: vxRT{target}}))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		opts []llms.CallOption
	}{
		{"logprobs+top", []llms.CallOption{llms.WithLogProbs(true), llms.WithTopLogProbs(1)}},
		{"top only", []llms.CallOption{llms.WithTopLogProbs(3)}},
	} {
		resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, c.opts...)
		var p map[string]any
		_ = json.Unmarshal(body, &p)
		delete(p, "messages")
		b, _ := json.Marshal(p)
		t.Logf("%s: err=%v request=%s", c.name, err, b)
		if resp != nil {
			ch, _ := json.Marshal(resp.Choices[0])
			t.Logf("%s: choice=%s", c.name, ch)
			keys := []string{}
			for k := range resp.Choices[0].GenerationInfo {
				keys = append(keys, k)
			}
			t.Logf("%s: genInfo keys=%v", c.name, keys)
		}
	}
}
