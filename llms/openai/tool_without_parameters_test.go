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

func TestAToolWithoutParametersOmitsThem(t *testing.T) {
	t.Parallel()

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"gpt-4.1",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"noon"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithBaseURL(srv.URL), WithToken("token"), WithModel("gpt-4.1"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time is it?")},
		llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
			Name: "clock", Description: "Tells the time",
		}}}))
	require.NoError(t, err)

	var sent struct {
		Tools []struct {
			Function map[string]json.RawMessage `json:"function"`
		} `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(raw, &sent))
	require.Len(t, sent.Tools, 1)
	assert.NotContains(t, sent.Tools[0].Function, "parameters")
	assert.JSONEq(t, `"clock"`, string(sent.Tools[0].Function["name"]))
}
