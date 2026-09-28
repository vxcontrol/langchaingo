package googleai

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

type f1rt struct{ body *string }

func (r f1rt) RoundTrip(req *http.Request) (*http.Response, error) {
	b, _ := io.ReadAll(req.Body)
	*r.body = string(b)
	resp := `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{}}`
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}},
		Body: io.NopCloser(bytes.NewBufferString(resp)), Request: req}, nil
}

func TestProbeF1ToolChoiceNoTools(t *testing.T) {
	for _, choice := range []any{"auto", "none"} {
		var body string
		llm, err := New(context.Background(), WithAPIKey("k"), WithHTTPClient(&http.Client{Transport: f1rt{&body}}),
			WithDefaultModel("gemini-2.5-flash"))
		if err != nil {
			t.Fatal(err)
		}
		_, err = llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithToolChoice(choice))
		t.Logf("choice=%v err=%v hasToolConfig=%v hasTools=%v", choice, err,
			strings.Contains(body, `"toolConfig"`), strings.Contains(body, `"tools"`))
		if i := strings.Index(body, `"toolConfig"`); i >= 0 {
			t.Logf("fragment: %s", body[i:min(len(body), i+70)])
		}
	}
}
