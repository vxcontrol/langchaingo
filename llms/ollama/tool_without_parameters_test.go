package ollama

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

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"glm-5","message":{"role":"assistant","content":"noon"},` +
			`"done":true,"done_reason":"stop"}` + "\n"))
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("glm-5"))
	require.NoError(t, err)

	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{Name: "clock", Description: "Tells the time"}}
	_, err = llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time is it?")},
		llms.WithTools([]llms.Tool{tool}))
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
	assert.Nil(t, tool.Function.Parameters, "the caller's definition must stay as it was")
}
