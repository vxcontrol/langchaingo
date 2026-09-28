package vertex

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/googleai"
)

type capT struct{ urls []string }

func (c *capT) RoundTrip(r *http.Request) (*http.Response, error) {
	c.urls = append(c.urls, r.URL.String())
	return nil, errors.New("captured")
}

func TestProbeV3F2(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	for _, ep := range []string{"us-central1-aiplatform.googleapis.com:443", "https://us-central1-aiplatform.googleapis.com"} {
		ct := &capT{}
		v, err := New(context.Background(), googleai.WithCloudProject("p"), googleai.WithCloudLocation("us-central1"),
			googleai.WithEndpoint(ep), googleai.WithHTTPClient(&http.Client{Transport: ct}), googleai.WithDefaultModel("gemini-2.0-flash"))
		t.Logf("ep=%s new err: %v", ep, err)
		if err == nil {
			_, gerr := v.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
			t.Logf("  urls=%v gen err=%v", ct.urls, gerr)
		}
	}
	v, err := New(context.Background(), googleai.WithCloudProject("p"), googleai.WithCloudLocation("us-central1"),
		googleai.WithEndpoint("us-central1-aiplatform.googleapis.com:443"), googleai.WithHTTPClient(&http.Client{}), googleai.WithDefaultModel("gemini-2.0-flash"))
	if err == nil {
		_, gerr := v.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
		t.Logf("plain client gen err=%v", gerr)
	}
	_, err = New(context.Background(), googleai.WithCloudProject("p"), googleai.WithCloudLocation("us-central1"), googleai.WithGRPCConn(nil))
	t.Logf("grpcconn err=%v", err)
}
