package huggingface

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

type probeRT struct {
	urls   []string
	bodies []string
}

func (p *probeRT) RoundTrip(r *http.Request) (*http.Response, error) {
	b, _ := io.ReadAll(r.Body)
	p.urls = append(p.urls, r.URL.String())
	p.bodies = append(p.bodies, string(b))
	return &http.Response{StatusCode: 404, Body: io.NopCloser(strings.NewReader("Not Found")), Header: http.Header{}, Request: r}, nil
}

func TestProbeProviderRoutes(t *testing.T) {
	for _, provider := range []string{"scaleway", "groq"} {
		rt := &probeRT{}
		llm, err := New(WithToken("t"), WithInferenceProvider(provider), WithModel("meta-llama/Llama-3.3-70B-Instruct"),
			WithHTTPClient(&http.Client{Transport: rt}))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
		_, err = llm.CreateEmbedding(context.Background(), []string{"hi"}, "Qwen/Qwen3-Embedding-8B", "feature-extraction")
		t.Logf("provider=%s urls=%v bodies=%v embErr=%v", provider, rt.urls, rt.bodies, err)
	}
}
