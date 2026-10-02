package bedrock_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

const warnSchema = `{"type":"object","properties":{"answer":{"type":"string"}},` +
	`"required":["answer"],"additionalProperties":false}`

const (
	legacySchemaAnswer = `{"id":"x","type":"message","role":"assistant","model":"m",` +
		`"content":[{"type":"text","text":"{\"answer\":\"ok\"}"}],"stop_reason":"end_turn",` +
		`"usage":{"input_tokens":1,"output_tokens":1}}`
	converseSchemaAnswer = `{"output":{"message":{"role":"assistant",` +
		`"content":[{"text":"{\"answer\":\"ok\"}"}]}},` +
		`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`
)

func TestASchemaThatReachesTheBedrockWireIsNotReportedAsLost(t *testing.T) {
	t.Parallel()

	for _, door := range []struct {
		name   string
		answer string
		opts   []bedrock.Option
	}{
		{"legacy", legacySchemaAnswer, []bedrock.Option{
			bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0"),
		}},
		{"converse", converseSchemaAnswer, []bedrock.Option{
			bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0"), bedrock.WithConverseAPI(),
		}},
	} {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			resp := bedrockWarningsFor(t, door.answer, door.opts,
				llms.WithStructuredOutput(llms.StructuredOutputConfig{
					Name: "s", Schema: json.RawMessage(warnSchema),
				}))

			require.NotContains(t, bedrockWarningsByOption(resp.Warnings), "WithJSONMode",
				"the schema reached the wire: %v", resp.Warnings)
		})
	}
}

func TestHaiku45ThroughTheIndiaProfileIsRefusedASchemaBeforeTheNetwork(t *testing.T) {
	t.Parallel()

	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "s", Schema: json.RawMessage(warnSchema)})
	for _, converse := range []bool{false, true} {
		answer := legacySchemaAnswer
		if converse {
			answer = converseSchemaAnswer
		}
		for _, model := range []string{
			"in.anthropic.claude-haiku-4-5-20251001-v1:0",
			"arn:aws:bedrock:ap-south-1:123456789012:inference-profile/in.anthropic.claude-haiku-4-5-20251001-v1:0",
		} {
			opts := []bedrock.Option{bedrock.WithModel(model)}
			if converse {
				opts = append(opts, bedrock.WithConverseAPI())
			}
			llm, sent := legacyLLMCapturing(t, answer, opts...)
			_, err := llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, schema)
			var refused *llms.ErrStructuredOutputUnsupported
			require.ErrorAs(t, err, &refused, "converse=%v %s", converse, model)
			require.Empty(t, *sent, "converse=%v %s: refused before the network", converse, model)
		}

		opts := []bedrock.Option{bedrock.WithModel("us.anthropic.claude-haiku-4-5-20251001-v1:0")}
		if converse {
			opts = append(opts, bedrock.WithConverseAPI())
		}
		llm, sent := legacyLLMCapturing(t, answer, opts...)
		_, err := llm.GenerateContent(t.Context(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, schema)
		require.NoError(t, err, "converse=%v: the us. profile serves the schema", converse)
		require.NotEmpty(t, *sent)
	}
}
