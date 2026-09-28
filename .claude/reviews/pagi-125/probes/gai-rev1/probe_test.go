package googleai_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/googleai"
)

type captureRT struct {
	mu     sync.Mutex
	urls   []string
	bodies []string
	auth   []string
}

func (c *captureRT) RoundTrip(r *http.Request) (*http.Response, error) {
	var b []byte
	if r.Body != nil {
		b, _ = io.ReadAll(r.Body)
	}
	c.mu.Lock()
	c.urls = append(c.urls, r.URL.String())
	c.bodies = append(c.bodies, string(b))
	c.auth = append(c.auth, r.Header.Get("Authorization")+"|"+r.Header.Get("x-goog-api-key"))
	c.mu.Unlock()
	body := `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{}}`
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewBufferString(body)),
		Request:    r,
	}, nil
}

const authorizedUserJSON = `{"type":"authorized_user","client_id":"id","client_secret":"secret","refresh_token":"tok"}`

func TestProbeCredsPlusExplicitKey(t *testing.T) {
	_, err := googleai.New(context.Background(),
		googleai.WithAPIKey("k"),
		googleai.WithCredentialsJSON([]byte(authorizedUserJSON)),
		googleai.WithHTTPClient(&http.Client{Transport: &captureRT{}}))
	fmt.Printf("PROBE creds+explicit key: err=%v\n", err)
}

func TestProbeCredsPlusEnvKey(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "envkey")
	rt := &captureRT{}
	llm, err := googleai.New(context.Background(),
		googleai.WithCredentialsJSON([]byte(authorizedUserJSON)),
		googleai.WithHTTPClient(&http.Client{Transport: rt}))
	fmt.Printf("PROBE creds+env key: New err=%v\n", err)
	if err == nil {
		_, err = llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
		fmt.Printf("PROBE creds+env key: Generate err=%v auth=%v\n", err, rt.auth)
	}
}

func TestProbeEndpointHostPort(t *testing.T) {
	rt := &captureRT{}
	llm, err := googleai.New(context.Background(),
		googleai.WithAPIKey("k"),
		googleai.WithHTTPClient(&http.Client{Transport: rt}),
		googleai.WithEndpoint("generativelanguage.googleapis.com:443"))
	fmt.Printf("PROBE endpoint host:port: New err=%v\n", err)
	if err != nil {
		return
	}
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	fmt.Printf("PROBE endpoint host:port: Generate err=%v urls=%v\n", err, rt.urls)
}

func TestProbeProLatestThinking(t *testing.T) {
	for _, model := range []string{"gemini-pro-latest", "gemini-2.5-pro", "gemini-flash-latest"} {
		rt := &captureRT{}
		llm, err := googleai.New(context.Background(),
			googleai.WithAPIKey("k"),
			googleai.WithHTTPClient(&http.Client{Transport: rt}))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithModel(model), llms.WithReasoning(llms.ReasoningHigh, 0))
		body := ""
		if len(rt.bodies) > 0 {
			body = rt.bodies[0]
		}
		i := strings.Index(body, `"thinkingConfig"`)
		tc := "<none>"
		if i >= 0 {
			tc = body[i:min(len(body), i+90)]
		}
		var warns any
		if resp != nil {
			warns = fmt.Sprintf("%+v", resp)
		}
		_ = warns
		fmt.Printf("PROBE %s WithReasoning(high): err=%v thinkingConfig=%s\n", model, err, tc)
	}
}

func TestProbeToolChoiceNoTools(t *testing.T) {
	rt := &captureRT{}
	llm, err := googleai.New(context.Background(),
		googleai.WithAPIKey("k"),
		googleai.WithHTTPClient(&http.Client{Transport: rt}))
	if err != nil {
		t.Fatal(err)
	}
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithToolChoice("auto"))
	fmt.Printf("PROBE toolchoice auto no tools: err=%v hasToolConfig=%v hasTools=%v\n", err,
		strings.Contains(rt.bodies[0], `"toolConfig"`), strings.Contains(rt.bodies[0], `"tools"`))
	if i := strings.Index(rt.bodies[0], `"toolConfig"`); i >= 0 {
		fmt.Printf("PROBE   %s\n", rt.bodies[0][i:min(len(rt.bodies[0]), i+80)])
	}
}
