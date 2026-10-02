package bedrock_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

type reflectedSchema struct {
	Type       string                    `json:"type"`
	Properties map[string]reflectedField `json:"properties"`
	Required   []string                  `json:"required"`
}

type reflectedField struct {
	Type     string   `json:"type"`
	Minimum  *int     `json:"minimum,omitempty"`
	MaxItems *uint64  `json:"maxItems,omitempty"`
	Maximum  *float64 `json:"maximum,omitempty"`
}

func TestConverseSendsAToolSchemaAsTheCallerWroteIt(t *testing.T) {
	t.Parallel()

	one, five, half := 1, uint64(5), 0.5
	schemas := map[string]any{
		"a map": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"items": map[string]any{"type": "array", "maxItems": 5, "items": map[string]any{"type": "string"}},
				"ratio": map[string]any{"type": "number", "minimum": 0.5},
			},
			"required": []string{"items"},
		},
		"raw JSON": json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer","maximum":5}},"required":["n"]}`),
		"a map of decoded numbers": map[string]any{
			"type":       "object",
			"properties": map[string]any{"n": map[string]any{"type": "integer", "maximum": json.Number("5")}},
		},
		"a map around raw JSON": map[string]any{
			"type":       "object",
			"properties": json.RawMessage(`{"n":{"type":"integer","minimum":1}}`),
		},
		"a struct": reflectedSchema{
			Type: "object",
			Properties: map[string]reflectedField{
				"count": {Type: "integer", Minimum: &one},
				"tags":  {Type: "array", MaxItems: &five},
				"share": {Type: "number", Maximum: &half},
			},
			Required: []string{"count"},
		},
	}
	for name, schema := range schemas {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			_, body := bedrockWarningsSending(t, converseAnswer,
				[]bedrock.Option{bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI()},
				llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
					Name: "lookup", Description: "Looks things up", Parameters: schema,
				}}}))

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
			encoded, err := json.Marshal(body)
			require.NoError(t, err)
			require.NoError(t, json.Unmarshal(encoded, &sent))
			require.Len(t, sent.ToolConfig.Tools, 1)

			want, err := json.Marshal(schema)
			require.NoError(t, err)
			require.JSONEq(t, string(want), string(sent.ToolConfig.Tools[0].ToolSpec.InputSchema.JSON))
		})
	}
}

type documentSchema struct {
	Type       string         `document:"type"`
	Properties documentFields `document:"properties"`
	Required   []string       `document:"required"`
}

type documentFields struct {
	Query documentField `document:"query"`
	Limit documentField `document:"limit"`
}

type documentField struct {
	Type    string `document:"type"`
	Maximum int    `document:"maximum,omitempty"`
}

func TestConverseKeepsTheFieldOrderOfASchemaTaggedForDocuments(t *testing.T) {
	t.Parallel()

	llm, sent := legacyLLMCapturing(t, converseAnswer,
		bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
	_, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
			Name: "search", Description: "Searches", Parameters: documentSchema{
				Type:       "object",
				Properties: documentFields{Query: documentField{Type: "string"}, Limit: documentField{Type: "integer", Maximum: 50}},
				Required:   []string{"query"},
			},
		}}}))
	require.NoError(t, err)
	require.Contains(t, *sent, `"json":{"type":"object","properties":{"query":{"type":"string"},`+
		`"limit":{"type":"integer","maximum":50}},"required":["query"]}`)
}

func TestConverseSendsAWideIntegerOfASchemaDigitForDigit(t *testing.T) {
	t.Parallel()

	llm, sent := legacyLLMCapturing(t, converseAnswer,
		bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
	_, err := llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
			Name: "lookup",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{
				"id": map[string]any{"type": "integer", "maximum": json.Number("9007199254740993")},
			}},
		}}}))
	require.NoError(t, err)
	require.Contains(t, *sent, `"maximum":9007199254740993`)
}
