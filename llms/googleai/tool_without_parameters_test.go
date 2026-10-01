package googleai

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

func TestAToolWithoutParametersIsDeclaredWithoutThem(t *testing.T) {
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
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"noon"}]},`+
					`"finishReason":"STOP"}]}`)
			}))
			t.Cleanup(server.Close)

			llm, err := New(t.Context(), WithAPIKey("unit-test-key"), WithEndpoint(server.URL),
				WithDefaultModel("gemini-2.5-flash"))
			require.NoError(t, err)

			_, err = llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time is it?")},
				llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
					Name: "clock", Description: "Tells the time", Parameters: parameters,
				}}}))
			require.NoError(t, err)

			var sent struct {
				Tools []struct {
					FunctionDeclarations []map[string]json.RawMessage `json:"functionDeclarations"`
				} `json:"tools"`
			}
			require.NoError(t, json.Unmarshal(raw, &sent))
			require.Len(t, sent.Tools, 1)
			require.Len(t, sent.Tools[0].FunctionDeclarations, 1)
			declaration := sent.Tools[0].FunctionDeclarations[0]
			assert.JSONEq(t, `"clock"`, string(declaration["name"]))
			assert.NotContains(t, declaration, "parameters")
			assert.NotContains(t, declaration, "parametersJsonSchema")
		})
	}
}
