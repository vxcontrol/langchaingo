package bedrock_test

import (
	"encoding/json"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestTheLegacyDoorKeepsTheAnswerLimitWithinTheModelsDocumentedMaximum(t *testing.T) {
	t.Parallel()

	const (
		titan    = `{"inputTextTokenCount":1,"results":[{"tokenCount":1,"outputText":"ok","completionReason":"FINISH"}]}`
		meta     = `{"generation":"ok","stop_reason":"stop","prompt_token_count":1,"generation_token_count":1}`
		cohere   = `{"generations":[{"text":"ok","finish_reason":"COMPLETE"}]}`
		commandR = `{"text":"ok","finish_reason":"COMPLETE","generation_id":"g"}`
		jamba    = `{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},` +
			`"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
		j2       = `{"completions":[{"data":{"text":"ok"},"finishReason":{"reason":"endoftext"}}]}`
		deepseek = `{"choices":[{"text":"ok","stop_reason":"stop"}]}`
	)
	for _, family := range []struct {
		model, answer string
		path          []string
		ceiling       int
	}{
		{"meta.llama3-70b-instruct-v1:0", meta, []string{"max_gen_len"}, 2048},
		{"us.meta.llama3-3-70b-instruct-v1:0", meta, []string{"max_gen_len"}, 2048},
		{"amazon.titan-text-express-v1", titan, []string{"textGenerationConfig", "maxTokenCount"}, 8192},
		{"amazon.titan-text-lite-v1", titan, []string{"textGenerationConfig", "maxTokenCount"}, 4096},
		{"amazon.titan-text-premier-v1:0", titan, []string{"textGenerationConfig", "maxTokenCount"}, 3072},
		{"us.amazon.nova-lite-v1:0", novaAnswer, []string{"inferenceConfig", "maxTokens"}, 5000},
		{"amazon.nova-2-lite-v1:0", novaAnswer, []string{"inferenceConfig", "maxTokens"}, 64000},
		{"cohere.command-text-v14", cohere, []string{"max_tokens"}, 4096},
		{"cohere.command-r-v1:0", commandR, []string{"max_tokens"}, 0},
		{"ai21.jamba-1-5-large-v1:0", jamba, []string{"max_tokens"}, 4096},
		{"ai21.j2-ultra-v1", j2, []string{"maxTokens"}, 8191},
		{"ai21.j2-grande-instruct", j2, []string{"maxTokens"}, 2048},
		{"deepseek.r1-v1:0", deepseek, []string{"max_tokens"}, 32768},
	} {
		t.Run(family.model, func(t *testing.T) {
			t.Parallel()

			limitSent := func(opts ...llms.CallOption) (float64, map[string]llms.Warning) {
				llm, sent := legacyLLMCapturing(t, family.answer, bedrock.WithModel(family.model))
				resp, err := llm.GenerateContent(t.Context(),
					[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
				require.NoError(t, err)
				var node any
				require.NoError(t, json.Unmarshal([]byte(*sent), &node))
				for _, key := range family.path {
					fields, _ := node.(map[string]any)
					node = fields[key]
				}
				limit, ok := node.(float64)
				require.True(t, ok, *sent)
				return limit, bedrockWarningsByOption(resp.Warnings)
			}

			unasked := llms.DefaultMaxTokens
			if family.ceiling != 0 {
				unasked = min(unasked, family.ceiling)
			}
			limit, warnings := limitSent()
			require.InDelta(t, unasked, limit, 0)
			require.NotContains(t, warnings, "WithMaxTokens", "the caller asked for nothing")

			limit, warnings = limitSent(llms.WithMaxTokens(1000))
			require.InDelta(t, 1000, limit, 0)
			require.NotContains(t, warnings, "WithMaxTokens")

			if family.ceiling == 0 {
				return
			}
			limit, warnings = limitSent(llms.WithMaxTokens(family.ceiling + 1))
			require.InDelta(t, family.ceiling, limit, 0)
			clamped, reported := warnings["WithMaxTokens"]
			require.True(t, reported)
			require.Equal(t, llms.WarningClamp, clamped.Kind)
			require.Equal(t, strconv.Itoa(family.ceiling+1), clamped.Asked)
			require.Equal(t, strconv.Itoa(family.ceiling), clamped.Sent)
		})
	}
}
