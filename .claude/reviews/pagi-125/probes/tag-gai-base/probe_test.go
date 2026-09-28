package googleai

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type probeEmpty struct{}

func (probeEmpty) RoundTrip(r *http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}},
		Body: io.NopCloser(bytes.NewReader(nil)), Request: r}, nil
}

func TestProbeEmptyStreamBase(t *testing.T) {
	llm, err := New(context.Background(), WithAPIKey("k"), WithRest(), WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: probeEmpty{}}))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	t.Logf("base: err=%v content=%q", err, func() string {
		if resp != nil {
			return resp.Choices[0].Content
		}
		return "<nil>"
	}())
}
