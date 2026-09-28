package mistral

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeMistralStructuredOutput(t *testing.T) {
	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"mistral-small-latest",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"{\"city\":\"Paris\"}"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)
	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	if err != nil {
		t.Fatal(err)
	}
	schema := json.RawMessage(`{"type":"object","properties":{"capital":{"type":"string"}},"required":["capital"],"additionalProperties":false}`)
	resp, err := m.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "capital of France")},
		llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "capital", Schema: schema}))
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	fmt.Printf("mistral structured output: err=%v content=%q response_format=%v\n", err, resp.Choices[0].Content, body["response_format"])
	for _, w := range resp.Warnings {
		fmt.Printf("warning: %s\n", w.String())
	}
}

func TestProbeMistralEndpointPaths(t *testing.T) {
	var mu sync.Mutex
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/mistral/v1/embeddings" || r.URL.Path == "/v1/embeddings" {
			_, _ = io.WriteString(w, `{"data":[{"embedding":[0.1]}]}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)
	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL+"/mistral"))
	if err != nil {
		t.Fatal(err)
	}
	_, err1 := m.Call(context.Background(), "hi")
	_, err2 := m.CreateEmbedding(context.Background(), []string{"x"})
	fmt.Printf("endpoint %s/mistral: chat err=%v embed err=%v paths=%v\n", "srv", err1, err2, paths)
}

func TestProbeMistralZeroRetries(t *testing.T) {
	var seen int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		atomic.AddInt32(&seen, 1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)
	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithMaxRetries(0))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = m.CreateEmbedding(context.Background(), []string{"x"})
	fmt.Println("WithMaxRetries(0) embeddings attempts:", atomic.LoadInt32(&seen))
	atomic.StoreInt32(&seen, 0)
	_, _ = m.Call(context.Background(), "hi")
	fmt.Println("WithMaxRetries(0) chat attempts:", atomic.LoadInt32(&seen))
}
