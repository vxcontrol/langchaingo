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

type vThoughtStream struct{}

func (vThoughtStream) RoundTrip(r *http.Request) (*http.Response, error) {
	const body = `data: {"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP","index":0}],` +
		`"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":7,"totalTokenCount":205,"thoughtsTokenCount":189}}` + "\r\n\r\n"
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}},
		Body: io.NopCloser(bytes.NewReader([]byte(body))), Request: r}, nil
}

func TestProbeVUsage(t *testing.T) {
	llm, err := New(context.Background(), WithAPIKey("k"), WithRest(), WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: vThoughtStream{}}))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	if err != nil {
		t.Fatal(err)
	}
	i := resp.Choices[0].GenerationInfo
	t.Logf("Prompt=%v Completion=%v Reasoning=%v Total=%v; Prompt+Completion=%d", i["PromptTokens"], i["CompletionTokens"], i["ReasoningTokens"], i["TotalTokens"], i["PromptTokens"].(int)+i["CompletionTokens"].(int))
}
