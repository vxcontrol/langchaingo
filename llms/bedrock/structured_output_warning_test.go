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
