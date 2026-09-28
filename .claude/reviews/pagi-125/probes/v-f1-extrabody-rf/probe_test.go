package openai_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func TestProbeExtraBodyRF(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		t.Logf("REQ: %s", b)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	rf := &openai.ResponseFormat{Type: "json_schema", JSONSchema: &openai.ResponseFormatJSONSchema{
		Name: "a", Strict: true,
		Schema: &openai.ResponseFormatJSONSchemaProperty{Type: "object",
			Properties: map[string]*openai.ResponseFormatJSONSchemaProperty{"city": {Type: "string"}},
			Required:   []string{"city"}},
	}}
	llm, err := openai.New(openai.WithToken("k"), openai.WithBaseURL(srv.URL), openai.WithModel("gpt-4o"), openai.WithResponseFormat(rf))
	if err != nil {
		t.Fatal(err)
	}
	extra := map[string]any{"response_format": map[string]any{"type": "json_schema", "json_schema": map[string]any{
		"name": "b", "strict": true,
		"schema": map[string]any{"type": "object", "additionalProperties": false,
			"properties": map[string]any{"score": map[string]any{"type": "number"}}, "required": []string{"score"}}}}}
	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, openai.WithExtraBody(extra))
	t.Logf("err=%v", err)
}
