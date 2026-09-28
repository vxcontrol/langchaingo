package httprr

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type v3rt struct{}

func (v3rt) RoundTrip(req *http.Request) (*http.Response, error) {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	h.Set("X-Litellm-Key-Spend", "12.3")
	h.Set("Llm_provider-Openai-Organization", "org-REALORG")
	h.Set("Llm_provider-Openai-Project", "proj_REALPROJECT")
	h.Set("Anthropic-Organization-Id", "1904456a-real")
	h.Set("Openai-Project", "proj_direct")
	return &http.Response{StatusCode: 200, Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1, Header: h,
		Body: io.NopCloser(strings.NewReader(`{}`)), ContentLength: 2, Request: req}, nil
}

func TestProbeV3LLMProv(t *testing.T) {
	path := filepath.Join(t.TempDir(), "p.httprr")
	defer setRecordForTesting(".*")()
	rr, err := Open(path, v3rt{})
	if err != nil {
		t.Fatal(err)
	}
	c := &http.Client{Transport: rr}
	resp, err := c.Post("https://api.example.com/v1/chat/completions", "application/json", bytes.NewReader([]byte(`{}`)))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	rr.Close()
	b, _ := os.ReadFile(path)
	t.Logf("\n%s", b)
}
