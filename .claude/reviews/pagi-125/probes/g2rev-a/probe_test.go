package googleai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type probeRT struct {
	status int
	ctype  string
	body   string
	url    string
	sent   []byte
}

func (p *probeRT) RoundTrip(r *http.Request) (*http.Response, error) {
	p.url = r.URL.String()
	if r.Body != nil {
		p.sent, _ = io.ReadAll(r.Body)
	}
	return &http.Response{
		StatusCode: p.status,
		Header:     http.Header{"Content-Type": []string{p.ctype}},
		Body:       io.NopCloser(bytes.NewReader([]byte(p.body))),
		Request:    r,
	}, nil
}

func TestProbeVertexListModels(t *testing.T) {
	rt := &probeRT{status: 200, ctype: "application/json",
		body: `{"publisherModels":[{"name":"publishers/google/models/gemini-2.5-flash","versionId":"001"},{"name":"publishers/google/models/gemini-3-pro-preview"}]}`}
	llm, err := New(context.Background(), WithCloudProject("p"), WithCloudLocation("us-central1"),
		WithHTTPClient(&http.Client{Transport: rt}))
	if err != nil {
		t.Fatal(err)
	}
	ids, err := llm.ListModels(context.Background())
	fmt.Printf("VERTEX URL=%s\nVERTEX IDS=%q err=%v\n", rt.url, ids, err)
}

func TestProbeEmptyStreamDefaultMaxTokens(t *testing.T) {
	rt := &probeRT{status: 200, ctype: "text/event-stream", body: ""}
	llm, err := New(context.Background(), WithAPIKey("k"), WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: rt}))
	if err != nil {
		t.Fatal(err)
	}
	sink := func(context.Context, streaming.Chunk) error { return nil }
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithStreamingFunc(sink))
	var e *llms.Error
	if errors.As(err, &e) {
		fmt.Printf("EMPTY STREAM (no WithMaxTokens): code=%v msg=%q\n", e.Code, e.Message)
	} else {
		fmt.Printf("EMPTY STREAM (no WithMaxTokens): err=%v\n", err)
	}
	fmt.Printf("maxOutputTokens sent: %s\n", extract(rt.sent, "maxOutputTokens"))
}

func extract(body []byte, key string) string {
	var m map[string]any
	_ = json.Unmarshal(body, &m)
	gc, _ := m["generationConfig"].(map[string]any)
	b, _ := json.Marshal(gc[key])
	return string(b)
}

func TestProbeGemma4Adaptive(t *testing.T) {
	for _, opt := range []llms.CallOption{
		llms.WithAdaptiveReasoning(llms.ReasoningNone),
		llms.WithAdaptiveReasoning(llms.ReasoningHigh),
		llms.WithReasoning(llms.ReasoningHigh, 0),
	} {
		rt := &probeRT{status: 200, ctype: "application/json",
			body: `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`}
		llm, err := New(context.Background(), WithAPIKey("k"), WithDefaultModel("gemma-4-31b-it"),
			WithHTTPClient(&http.Client{Transport: rt}))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opt)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("GEMMA4 thinkingConfig=%s warnings=%+v\n", extract(rt.sent, "thinkingConfig"), resp.Warnings)
	}
}
