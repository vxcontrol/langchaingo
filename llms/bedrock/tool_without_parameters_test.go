package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

var noParameters = map[string]any{
	"nil":             nil,
	"nil map":         map[string]any(nil),
	"nil raw message": json.RawMessage(nil),
	"null":            json.RawMessage("null"),
}

func clockWithParameters(parameters any) llms.CallOption {
	return llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "clock", Description: "Tells the time", Parameters: parameters,
	}}})
}

func TestAConverseToolWithoutParametersSendsAnEmptyObjectSchema(t *testing.T) {
	t.Parallel()

	for name, parameters := range noParameters {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			var raw []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"output":{"message":{"role":"assistant","content":[{"text":"noon"}]}},`+
					`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
			}))
			t.Cleanup(srv.Close)

			llm := bedrockLLMAgainst(t, srv,
				bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())

			_, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time is it?")},
				clockWithParameters(parameters))
			require.NoError(t, err)

			var sent struct {
				ToolConfig struct {
					Tools []struct {
						ToolSpec struct {
							InputSchema struct {
								JSON json.RawMessage `json:"json"`
							} `json:"inputSchema"`
						} `json:"toolSpec"`
					} `json:"tools"`
				} `json:"toolConfig"`
			}
			require.NoError(t, json.Unmarshal(raw, &sent))
			require.Len(t, sent.ToolConfig.Tools, 1)
			assert.JSONEq(t, `{"type":"object","properties":{}}`,
				string(sent.ToolConfig.Tools[0].ToolSpec.InputSchema.JSON))
		})
	}
}

func TestALegacyClaudeToolWithoutParametersSendsAnEmptyObjectSchema(t *testing.T) {
	t.Parallel()

	for name, parameters := range noParameters {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			llm, body := legacyLLMCapturing(t, `{"id":"x","type":"message","role":"assistant","model":"m",`+
				`"content":[{"type":"text","text":"noon"}],"stop_reason":"end_turn",`+
				`"usage":{"input_tokens":1,"output_tokens":1}}`,
				bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"))

			_, err := llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time is it?")},
				clockWithParameters(parameters))
			require.NoError(t, err)

			var sent struct {
				Tools []struct {
					InputSchema json.RawMessage `json:"input_schema"`
				} `json:"tools"`
			}
			require.NoError(t, json.Unmarshal([]byte(*body), &sent))
			require.Len(t, sent.Tools, 1)
			assert.JSONEq(t, `{"type":"object","properties":{}}`, string(sent.Tools[0].InputSchema))
		})
	}
}
