package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
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
		"us.openai.gpt-5.6-luna": true, "openai.gpt-6.2-sol": true, "us.openai.gpt-7-sol": true,
		"moonshotai.kimi-k3": false, "us.xai.grok-4.7": false, "openai.gpt-oss-120b-1:0": false,
		"moonshotai.kimi-k4": false, "xai.grok-4.8": false,
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

func TestAStreamedSchemaReachesTheModelsWhoseCardsDocumentItOnConverseStream(t *testing.T) {
	t.Parallel()

	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "s", Schema: json.RawMessage(warnSchema)})
	for _, model := range []string{"us.openai.gpt-6.1-sol", "global.openai.gpt-6.1-sol", "openai.gpt-6.2-sol", "us.openai.gpt-7-sol"} {
		resp, body := converseStreamSending(t, model, schema)
		require.Contains(t, body, "outputConfig", model)
		extra, _ := body["additionalModelRequestFields"].(map[string]any)
		require.Equal(t, map[string]any{"format": map[string]any{"strict": true}}, extra["text"], model)
		_, warned := bedrockWarningsByOption(resp.Warnings)["WithStructuredOutput"]
		require.False(t, warned, "%s: %v", model, resp.Warnings)
	}
}

func converseStreamSending(t *testing.T, model string, call ...llms.CallOption) (*llms.ContentResponse, map[string]any) {
	t.Helper()

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		enc := eventstream.NewEncoder()
		writeConverseEvent(t, w, enc, "messageStart", `{"role":"assistant"}`)
		writeConverseEvent(t, w, enc, "contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"{\"answer\":\"ok\"}"}}`)
		writeConverseEvent(t, w, enc, "contentBlockStop", `{"contentBlockIndex":0}`)
		writeConverseEvent(t, w, enc, "messageStop", `{"stopReason":"end_turn"}`)
		writeConverseEvent(t, w, enc, "metadata", `{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, bedrock.WithModel(model), bedrock.WithConverseAPI())
	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		append([]llms.CallOption{llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil })}, call...)...)
	require.NoError(t, err, model)
	var body map[string]any
	require.NoError(t, json.Unmarshal(raw, &body), model)
	return resp, body
}

func requireInheritedSchema(t *testing.T, resp *llms.ContentResponse, model string) {
	t.Helper()

	w := bedrockWarningsByOption(resp.Warnings)["WithStructuredOutput"]
	require.Equal(t, llms.WarningInherit, w.Kind, "%s: %v", model, resp.Warnings)
	require.Equal(t, "s", w.Asked, model)
	require.Equal(t, "s", w.Sent, model)
}

func TestAnUnlistedVersionOnBedrockIsSentASchemaItsReleaseWouldRefuse(t *testing.T) {
	t.Parallel()

	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "s", Schema: json.RawMessage(warnSchema)})
	for _, model := range []string{"anthropic.claude-opus-6-0-v1:0", "us.anthropic.claude-sonnet-6-v1:0", "zai.glm-6"} {
		resp, body := bedrockWarningsSending(t, converseSchemaAnswer,
			[]bedrock.Option{bedrock.WithModel(model), bedrock.WithConverseAPI()}, schema)
		format, _ := body["outputConfig"].(map[string]any)
		require.Contains(t, format, "textFormat", "%s: %v", model, body)
		requireInheritedSchema(t, resp, model)
	}

	resp, body := bedrockWarningsSending(t, legacySchemaAnswer,
		[]bedrock.Option{bedrock.WithModel("anthropic.claude-opus-6-0-v1:0")}, schema)
	config, _ := body["output_config"].(map[string]any)
	format, _ := config["format"].(map[string]any)
	require.Equal(t, "json_schema", format["type"], "%v", body)
	requireInheritedSchema(t, resp, "legacy anthropic.claude-opus-6-0-v1:0")

	for _, model := range []string{"us.openai.gpt-6.1-luna", "openai.gpt-5.7-sol", "openai.gpt-6.1-astra"} {
		resp, body := converseStreamSending(t, model, schema)
		require.Contains(t, body, "outputConfig", model)
		extra, _ := body["additionalModelRequestFields"].(map[string]any)
		require.Equal(t, map[string]any{"format": map[string]any{"strict": true}}, extra["text"], model)
		requireInheritedSchema(t, resp, model)
	}
}

func TestAListedReleaseOnBedrockKeepsItsSchemaRefusal(t *testing.T) {
	t.Parallel()

	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "s", Schema: json.RawMessage(warnSchema)})
	for _, door := range []struct {
		answer string
		opts   []bedrock.Option
	}{
		{legacySchemaAnswer, nil},
		{converseSchemaAnswer, []bedrock.Option{bedrock.WithConverseAPI()}},
	} {
		for _, model := range []string{"anthropic.claude-opus-5-5-v1:0", "us.anthropic.claude-opus-4-7-v1:0"} {
			llm, sent := legacyLLMCapturing(t, door.answer, append([]bedrock.Option{bedrock.WithModel(model)}, door.opts...)...)
			_, err := llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, schema)
			var refused *llms.ErrStructuredOutputUnsupported
			require.ErrorAs(t, err, &refused, model)
			require.Contains(t, refused.Reason, "neither API", model)
			require.Empty(t, *sent, "%s: refused before the network", model)
		}
	}
}
