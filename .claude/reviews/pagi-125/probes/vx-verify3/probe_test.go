package vertex_test

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/googleai/vertex"
)

const creds = `{"type":"authorized_user","client_id":"id.apps.googleusercontent.com","client_secret":"secret","refresh_token":"token"}`

type capRT struct {
	mu   sync.Mutex
	reqs []string
}

func (c *capRT) RoundTrip(r *http.Request) (*http.Response, error) {
	c.mu.Lock()
	c.reqs = append(c.reqs, r.URL.Host+" auth="+r.Header.Get("Authorization"))
	c.mu.Unlock()
	body := `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
}

func TestProbeHTTPClientCreds(t *testing.T) {
	for _, withRest := range []bool{false, true} {
		rt := &capRT{}
		opts := []googleai.Option{googleai.WithCloudProject("p"), googleai.WithCloudLocation("us-central1"),
			googleai.WithCredentialsJSON([]byte(creds)), googleai.WithHTTPClient(&http.Client{Transport: rt})}
		if withRest {
			opts = append(opts, googleai.WithRest())
		}
		v, err := vertex.New(context.Background(), opts...)
		t.Logf("withRest=%v new err: %v", withRest, err)
		if err != nil {
			continue
		}
		_, err = v.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithModel("gemini-2.5-flash"))
		t.Logf("withRest=%v gen err: %v requests: %v", withRest, err, rt.reqs)
	}
}

func TestProbeLocation(t *testing.T) {
	for _, k := range []string{"GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION", "CLOUD_ML_REGION", "GOOGLE_CLOUD_PROJECT"} {
		t.Setenv(k, "")
	}
	_, err := vertex.New(context.Background(), googleai.WithCloudProject("p"), googleai.WithCredentialsJSON([]byte(creds)))
	t.Logf("no location err: %v", err)
	t.Setenv("GOOGLE_CLOUD_REGION", "europe-west4")
	rt := &capRT{}
	v, err := vertex.New(context.Background(), googleai.WithCloudProject("p"), googleai.WithCredentialsJSON([]byte(creds)), googleai.WithHTTPClient(&http.Client{Transport: rt}), googleai.WithRest())
	t.Logf("region env err: %v", err)
	if err == nil {
		_, err = v.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithModel("gemini-2.5-flash"))
		t.Logf("region env gen err: %v reqs %v", err, rt.reqs)
	}
}
