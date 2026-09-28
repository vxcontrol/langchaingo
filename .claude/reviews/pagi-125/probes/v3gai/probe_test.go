package googleai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/googleai"
)

const saJSON = `{"type":"authorized_user","client_id":"id.apps.googleusercontent.com","client_secret":"secret","refresh_token":"token"}`

type capRT struct {
	urls   []string
	bodies []string
}

func (c *capRT) RoundTrip(r *http.Request) (*http.Response, error) {
	c.urls = append(c.urls, r.URL.String()+" key="+r.Header.Get("x-goog-api-key"))
	if r.Body != nil {
		b, _ := io.ReadAll(r.Body)
		c.bodies = append(c.bodies, string(b))
	}
	resp := `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(bytes.NewBufferString(resp)), Request: r}, nil
}

func call(t *testing.T, opts ...googleai.Option) (*capRT, error) {
	rt := &capRT{}
	opts = append(opts, googleai.WithHTTPClient(&http.Client{Transport: rt}))
	c, err := googleai.New(context.Background(), opts...)
	if err != nil {
		return rt, err
	}
	_, err = c.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	return rt, err
}

func TestProbeCredsExplicitKey(t *testing.T) {
	rt, err := call(t, googleai.WithAPIKey("k"), googleai.WithCredentialsJSON([]byte(saJSON)))
	t.Logf("explicit key + creds JSON: err=%v urls=%v", err, rt.urls)
}

func TestProbeCredsEnvKey(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "envkey")
	path := filepath.Join(t.TempDir(), "c.json")
	os.WriteFile(path, []byte(saJSON), 0o600)
	rt, err := call(t, googleai.WithCredentialsFile(path))
	t.Logf("env key + creds file: err=%v urls=%v", err, rt.urls)
}

func TestProbeEndpointHostPort(t *testing.T) {
	rt, err := call(t, googleai.WithAPIKey("k"), googleai.WithEndpoint("generativelanguage.googleapis.com:443"))
	t.Logf("host:port endpoint: err=%v urls=%v", err, rt.urls)
	rt, err = call(t, googleai.WithAPIKey("k"), googleai.WithEndpoint("https://generativelanguage.googleapis.com/"))
	t.Logf("full URL endpoint: err=%v urls=%v", err, rt.urls)
}

func TestProbeProLatest(t *testing.T) {
	for _, m := range []string{"gemini-pro-latest", "gemini-flash-latest", "gemini-2.5-pro"} {
		rt := &capRT{}
		c, err := googleai.New(context.Background(), googleai.WithAPIKey("k"), googleai.WithHTTPClient(&http.Client{Transport: rt}))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := c.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithModel(m), llms.WithReasoning(llms.ReasoningHigh, 0))
		tc := "<none>"
		if len(rt.bodies) > 0 {
			var body map[string]any
			json.Unmarshal([]byte(rt.bodies[0]), &body)
			if gc, ok := body["generationConfig"].(map[string]any); ok {
				if th, ok := gc["thinkingConfig"]; ok {
					b, _ := json.Marshal(th)
					tc = string(b)
				}
			}
		}
		warn := ""
		if resp != nil {
			b, _ := json.Marshal(resp)
			if strings.Contains(string(b), "does not think") {
				warn = " WARNING: does not think"
			}
		}
		t.Logf("%s: err=%v thinkingConfig=%s%s", m, err, tc, warn)
	}
}
