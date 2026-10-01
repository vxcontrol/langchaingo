package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAToolWithoutParametersSendsAnEmptyObjectSchema(t *testing.T) {
	t.Parallel()

	const empty = `{"type":"object","properties":{}}`
	for name, tc := range map[string]struct {
		parameters any
		strict     bool
		want       string
	}{
		"nil":             {parameters: nil, want: empty},
		"nil map":         {parameters: map[string]any(nil), want: empty},
		"nil raw message": {parameters: json.RawMessage(nil), want: empty},
		"null":            {parameters: json.RawMessage("null"), want: empty},
		"strict": {
			strict: true,
			want:   `{"type":"object","properties":{},"additionalProperties":false,"required":[]}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			var raw []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"mistral-large-latest",`+
					`"choices":[{"index":0,"message":{"role":"assistant","content":"noon"},"finish_reason":"stop"}],`+
					`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			}))
			t.Cleanup(srv.Close)

			llm, err := New(WithBaseURL(srv.URL), WithToken("token"), WithModel("mistral-large-latest"))
			require.NoError(t, err)

			_, err = llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time is it?")},
				llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
					Name: "clock", Description: "Tells the time", Parameters: tc.parameters, Strict: tc.strict,
				}}}))
			require.NoError(t, err)

			var sent struct {
				Tools []struct {
					Function struct {
						Parameters json.RawMessage `json:"parameters"`
					} `json:"function"`
				} `json:"tools"`
			}
			require.NoError(t, json.Unmarshal(raw, &sent))
			require.Len(t, sent.Tools, 1)
			assert.JSONEq(t, tc.want, string(sent.Tools[0].Function.Parameters))
		})
	}
}
