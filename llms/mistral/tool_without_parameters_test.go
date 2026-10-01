package mistral

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

	for name, parameters := range map[string]any{
		"nil":             nil,
		"nil map":         map[string]any(nil),
		"nil raw message": json.RawMessage(nil),
		"null":            json.RawMessage("null"),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			var raw []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"mistral-small-latest",`+
					`"choices":[{"index":0,"message":{"role":"assistant","content":"noon"},"finish_reason":"stop"}],`+
					`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			}))
			t.Cleanup(srv.Close)

			m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
			require.NoError(t, err)

			_, err = m.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time is it?")},
				llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
					Name: "clock", Description: "Tells the time", Parameters: parameters,
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
			assert.JSONEq(t, `{"type":"object","properties":{}}`, string(sent.Tools[0].Function.Parameters))
		})
	}
}
