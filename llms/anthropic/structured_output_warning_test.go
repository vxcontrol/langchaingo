package anthropic_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
)

func TestASchemaThatReachesTheAnthropicWireIsNotReportedAsLost(t *testing.T) {
	t.Parallel()

	const schema = `{"type":"object","properties":{"answer":{"type":"string"}},` +
		`"required":["answer"],"additionalProperties":false}`

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"m","type":"message","role":"assistant",`+
			`"model":"claude-sonnet-4-5","content":[{"type":"text","text":"{\"answer\":\"ok\"}"}],`+
			`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(
		anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL),
		anthropic.WithModel("claude-sonnet-4-5"),
	)
	require.NoError(t, err)

	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStructuredOutput(llms.StructuredOutputConfig{
			Name: "s", Schema: json.RawMessage(schema),
		}))
	require.NoError(t, err)

	require.NotContains(t, warningsByOption(resp.Warnings), "WithJSONMode",
		"the schema reached the wire: %v", resp.Warnings)
}
