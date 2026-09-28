package vertex_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/googleai/vertex"
)

type vfListRT struct{ url string }

func (p *vfListRT) RoundTrip(r *http.Request) (*http.Response, error) {
	p.url = r.URL.String()
	body := `{"publisherModels":[{"name":"publishers/google/models/gemini-2.5-flash"},{"name":"publishers/google/models/gemini-3-pro-preview"}]}`
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(bytes.NewReader([]byte(body))), Request: r}, nil
}

func TestProbeVFList(t *testing.T) {
	rt := &vfListRT{}
	v, err := vertex.New(context.Background(), googleai.WithCloudProject("p"), googleai.WithCloudLocation("us-central1"),
		googleai.WithHTTPClient(&http.Client{Transport: rt}))
	if err != nil {
		t.Fatal(err)
	}
	ids, err := v.ListModels(context.Background())
	fmt.Printf("PROBE URL=%s\nPROBE IDS=%q err=%v\n", rt.url, ids, err)
}
