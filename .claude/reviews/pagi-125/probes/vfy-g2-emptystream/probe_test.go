package googleai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type vfyRT struct{ sent []byte }

func (p *vfyRT) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.Body != nil {
		p.sent, _ = io.ReadAll(r.Body)
	}
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}},
		Body: io.NopCloser(bytes.NewReader(nil)), Request: r}, nil
}

func TestProbeVfyEmptyStream(t *testing.T) {
	rt := &vfyRT{}
	llm, err := New(context.Background(), WithAPIKey("k"), WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: rt}))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	var e *llms.Error
	if errors.As(err, &e) {
		fmt.Printf("RESULT code=%v msg=%q\n", e.Code, e.Message)
	} else {
		fmt.Printf("RESULT err=%v resp=%v\n", err, resp != nil)
	}
	fmt.Printf("SENT %s\n", rt.sent)
}
