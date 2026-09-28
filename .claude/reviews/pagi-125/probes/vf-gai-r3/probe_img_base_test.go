package googleai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

type vfImgRT struct{ sent []byte }

func (p *vfImgRT) RoundTrip(r *http.Request) (*http.Response, error) {
	p.sent, _ = io.ReadAll(r.Body)
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}},
		Body:    io.NopCloser(bytes.NewReader([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`))),
		Request: r}, nil
}

func TestProbeVFImage(t *testing.T) {
	for _, model := range []string{"gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview", "gemini-2.5-flash-image"} {
		for _, name := range []string{"on-high", "off"} {
			opt := llms.WithReasoning(llms.ReasoningHigh, 0)
			if name == "off" {
				opt = llms.WithReasoningDisabled()
			}
			rt := &vfImgRT{}
			llm, err := New(context.Background(), WithAPIKey("k"), WithDefaultModel(model), WithHTTPClient(&http.Client{Transport: rt}))
			if err != nil {
				t.Fatal(err)
			}
			resp, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "draw a cat")}, opt)
			var m map[string]any
			_ = json.Unmarshal(rt.sent, &m)
			gc, _ := m["generationConfig"].(map[string]any)
			b, _ := json.Marshal(gc["thinkingConfig"])
			fmt.Printf("PROBE %s %s: err=%v thinkingConfig=%s\n", model, name, err, b)
			_ = resp
		}
	}
}
