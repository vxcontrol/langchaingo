package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestThinkingNoLegacyPayloadCarriesIsReported(t *testing.T) {
	t.Parallel()

	const novaAnswer = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},` +
		`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`
	const metaAnswer = `{"generation":"ok","stop_reason":"stop","prompt_token_count":1,` +
		`"generation_token_count":1}`

	for _, door := range []struct {
		name   string
		model  string
		answer string
	}{
		{"nova without reasoning", "amazon.nova-lite-v1:0", novaAnswer},
		{"a payload with no thinking field at all", "meta.llama3-8b-instruct-v1:0", metaAnswer},
	} {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			resp := bedrockWarningsFor(t, door.answer,
				[]bedrock.Option{bedrock.WithModel(door.model)},
				llms.WithReasoning(llms.ReasoningHigh, 0))

			w, ok := bedrockWarningsByOption(resp.Warnings)["WithReasoning"]
			require.True(t, ok, "the ask to think went nowhere unreported: %v", resp.Warnings)
			require.Equal(t, llms.WarningDrop, w.Kind)
			require.Equal(t, "high", w.Asked)
		})
	}
}

func TestANovaModelThatThinksReportsNoLostThinking(t *testing.T) {
	t.Parallel()

	const novaAnswer = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},` +
		`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`

	resp := bedrockWarningsFor(t, novaAnswer,
		[]bedrock.Option{bedrock.WithModel("amazon.nova-2-lite-v1:0")},
		llms.WithReasoning(llms.ReasoningHigh, 0))

	w, ok := bedrockWarningsByOption(resp.Warnings)["WithReasoning"]
	if ok {
		require.NotEqual(t, llms.WarningDrop, w.Kind, "the effort reached the wire: %v", resp.Warnings)
	}
}
