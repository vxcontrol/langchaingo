package bedrock_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestTheLegacyDoorSendsAnExplicitZeroTemperatureAndNoneWhenUnasked(t *testing.T) {
	t.Parallel()

	for _, family := range []struct {
		model  string
		answer string
		path   []string
	}{
		{"amazon.titan-text-express-v1",
			`{"inputTextTokenCount":1,"results":[{"tokenCount":1,"outputText":"ok","completionReason":"FINISH"}]}`,
			[]string{"textGenerationConfig", "temperature"}},
		{"ai21.j2-ultra-v1",
			`{"completions":[{"data":{"text":"ok"},"finishReason":{"reason":"endoftext"}}]}`,
			[]string{"temperature"}},
		{"ai21.jamba-1-5-large-v1:0",
			`{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
				`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			[]string{"temperature"}},
		{"cohere.command-text-v14", `{"generations":[{"text":"ok","finish_reason":"COMPLETE"}]}`, []string{"temperature"}},
		{"cohere.command-r-v1:0", `{"text":"ok","finish_reason":"COMPLETE","generation_id":"g"}`, []string{"temperature"}},
		{"meta.llama3-70b-instruct-v1:0",
			`{"generation":"ok","stop_reason":"stop","prompt_token_count":1,"generation_token_count":1}`,
			[]string{"temperature"}},
		{"deepseek.r1-v1:0", `{"choices":[{"text":"ok","stop_reason":"stop"}]}`, []string{"temperature"}},
		{"amazon.nova-2-lite-v1:0", novaAnswer, []string{"inferenceConfig", "temperature"}},
	} {
		t.Run(family.model, func(t *testing.T) {
			t.Parallel()

			temperatureSent := func(opts ...llms.CallOption) (any, bool) {
				llm, sent := legacyLLMCapturing(t, family.answer, bedrock.WithModel(family.model))
				_, err := llm.GenerateContent(t.Context(),
					[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
				require.NoError(t, err)
				var node any
				require.NoError(t, json.Unmarshal([]byte(*sent), &node))
				for _, key := range family.path {
					fields, ok := node.(map[string]any)
					require.True(t, ok, *sent)
					if node, ok = fields[key]; !ok {
						return nil, false
					}
				}
				return node, true
			}

			value, sent := temperatureSent(llms.WithTemperature(0))
			require.True(t, sent, "an explicit 0 is a value the vendor takes")
			require.InDelta(t, 0, value, 0)

			_, sent = temperatureSent()
			require.False(t, sent, "an unasked temperature leaves the vendor's default")
		})
	}
}
