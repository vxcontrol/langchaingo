package bedrock_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
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
			require.Contains(t, refused.Reason, "other inference profiles", "converse=%v %s", converse, model)
			require.Empty(t, *sent, "converse=%v %s: refused before the network", converse, model)
		}

		for _, model := range []string{
			"us.anthropic.claude-haiku-4-5-20251001-v1:0", "in.anthropic.claude-sonnet-4-5-20250929-v1:0",
		} {
			opts := []bedrock.Option{bedrock.WithModel(model)}
			if converse {
				opts = append(opts, bedrock.WithConverseAPI())
			}
			llm, sent := legacyLLMCapturing(t, answer, opts...)
			_, err := llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, schema)
			require.NoError(t, err, "converse=%v %s: the profile serves the schema", converse, model)
			require.NotEmpty(t, *sent)
		}
	}
}

func TestTheModelsWhoseCardsListAConverseSchemaAreSentOne(t *testing.T) {
	t.Parallel()

	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "s", Schema: json.RawMessage(warnSchema)})
	for model, strict := range map[string]bool{
		"openai.gpt-6-sol": true, "us.openai.gpt-5.6-sol": true, "global.openai.gpt-6.1-sol": true,
		"in.openai.gpt-5.6-terra": true, "openai.gpt-6-astra": true, "openai.gpt-6-luna": true,
		"us.openai.gpt-5.6-luna": true,
		"moonshotai.kimi-k3":     false, "us.xai.grok-4.7": false, "openai.gpt-oss-120b-1:0": false,
	} {
		_, body := bedrockWarningsSending(t, converseSchemaAnswer,
			[]bedrock.Option{bedrock.WithModel(model), bedrock.WithConverseAPI()}, schema)
		require.Contains(t, body, "outputConfig", model)
		extra, _ := body["additionalModelRequestFields"].(map[string]any)
		if !strict {
			require.NotContains(t, extra, "text", model)
			continue
		}
		require.Equal(t, map[string]any{"format": map[string]any{"strict": true}}, extra["text"], model)
	}
}

func TestAStreamedSchemaIsRefusedWhereTheCardDocumentsItForNonStreamingCallsOnly(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"us.openai.gpt-6-sol", "global.openai.gpt-5.6-luna"} {
		llm, sent := legacyLLMCapturing(t, converseSchemaAnswer, bedrock.WithModel(model), bedrock.WithConverseAPI())
		_, err := llm.GenerateContent(t.Context(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "s", Schema: json.RawMessage(warnSchema)}),
			llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
		var refused *llms.ErrStructuredOutputUnsupported
		require.ErrorAs(t, err, &refused, model)
		require.Contains(t, refused.Reason, "non-streaming", model)
		require.Empty(t, *sent, "%s: refused before the network", model)
	}

	llm, sent := legacyLLMCapturing(t, converseSchemaAnswer,
		bedrock.WithModel("us.openai.gpt-6-sol"), bedrock.WithConverseAPI())

	_, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "s", Schema: json.RawMessage(warnSchema)}))
	require.NoError(t, err, "the same model takes the schema on a non-streaming call")
	require.NotEmpty(t, *sent)
}
