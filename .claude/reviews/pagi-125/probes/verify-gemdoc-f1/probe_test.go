package vertex_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/googleai/vertex"
)

type probeRT struct{ host string }

func (t *probeRT) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = "http"
	req.URL.Host = t.host
	return http.DefaultTransport.RoundTrip(req)
}

func TestProbeVertexOff(t *testing.T) {
	for _, m := range []string{"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"} {
		var body string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			body = string(b)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`)
		}))
		llm, err := vertex.New(context.Background(),
			googleai.WithCloudProject("p"), googleai.WithCloudLocation("us-central1"),
			googleai.WithDefaultModel(m),
			googleai.WithHTTPClient(&http.Client{Transport: &probeRT{host: srv.Listener.Addr().String()}}))
		if err != nil {
			t.Fatal(err)
		}
		_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithReasoningDisabled())
		t.Logf("model=%s gen err: %v\nbody=%s", m, err, body)
		srv.Close()
	}
}
