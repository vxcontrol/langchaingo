package googleai

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type probeThinkingStream struct{}

func (probeThinkingStream) RoundTrip(r *http.Request) (*http.Response, error) {
	const body = `data: {"candidates":[{"content":{"parts":[{"text":"The water cycle, also known as"}],"role":"model"},` +
		`"finishReason":"MAX_TOKENS","index":0}],"usageMetadata":{"promptTokenCount":9,` +
		`"candidatesTokenCount":7,"totalTokenCount":205,"thoughtsTokenCount":189}}` + "\r\n\r\n"
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}},
		Body: io.NopCloser(bytes.NewReader([]byte(body))), Request: r}, nil
}

func TestProbeEmptyStreamDefaultMaxTokens(t *testing.T) {
	llm, err := New(context.Background(), WithAPIKey("k"), WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: emptyStream{}}))
	if err != nil {
		t.Fatal(err)
	}
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	var apiErr *llms.Error
	if errors.As(err, &apiErr) {
		t.Logf("no WithMaxTokens (default %d): code=%v msg=%q", DefaultOptions().DefaultMaxTokens, apiErr.Code, apiErr.Message)
	} else {
		t.Logf("no WithMaxTokens: err=%v", err)
	}
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithMaxTokens(65536),
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	if errors.As(err, &apiErr) {
		t.Logf("WithMaxTokens(65536): code=%v msg=%q", apiErr.Code, apiErr.Message)
	}
}

func TestProbeStreamedTotalsWithThoughts(t *testing.T) {
	llm, err := New(context.Background(), WithAPIKey("k"), WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: probeThinkingStream{}}))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	if err != nil {
		t.Fatal(err)
	}
	info := resp.Choices[0].GenerationInfo
	t.Logf("Prompt=%v Completion=%v Reasoning=%v Total=%v; Prompt+Completion=%d",
		info["PromptTokens"], info["CompletionTokens"], info["ReasoningTokens"], info["TotalTokens"],
		info["PromptTokens"].(int)+info["CompletionTokens"].(int))
}
