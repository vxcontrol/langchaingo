package mistral

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeVrfxMistralSO(t *testing.T) {
	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"mistral-small-latest","choices":[{"index":0,"message":{"role":"assistant","content":"{\"city\":\"Paris\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	if err != nil {
		t.Fatal(err)
	}
	for _, schema := range []string{
		`{"type":"object","properties":{"capital":{"type":"string"}},"required":["capital"],"additionalProperties":false}`,
		`[1,2]`,
	} {
		resp, err := m.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "capital of France")},
			llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "capital", Schema: json.RawMessage(schema)}))
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		c := ""
		if resp != nil {
			c = resp.Choices[0].Content
		}
		fmt.Printf("schema=%s err=%v content=%q response_format=%v\n", schema, err, c, body["response_format"])
		if resp != nil {
			for _, w := range resp.Warnings {
				fmt.Println("  warning:", w.String())
			}
		}
	}
}
