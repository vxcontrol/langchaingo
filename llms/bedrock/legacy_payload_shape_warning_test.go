package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestOneProviderWithTwoPayloadShapesIsReportedByShape(t *testing.T) {
	t.Parallel()

	const jambaAnswer = `{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},` +
		`"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	const commandRAnswer = `{"text":"ok","finish_reason":"COMPLETE","generation_id":"g"}`

	for _, shape := range []struct {
		name    string
		model   string
		answer  string
		lost    []string
		carried []string
	}{
		{
			name: "jamba has no penalty field", model: "ai21.jamba-1-5-large-v1:0", answer: jambaAnswer,
			lost:    []string{"WithFrequencyPenalty", "WithPresencePenalty", "WithRepetitionPenalty"},
			carried: []string{"WithCandidateCount"},
		},
		{
			name: "command-r has no num_generations", model: "cohere.command-r-v1:0", answer: commandRAnswer,
			lost:    []string{"WithCandidateCount"},
			carried: nil,
		},
	} {
		t.Run(shape.name, func(t *testing.T) {
			t.Parallel()

			resp := bedrockWarningsFor(t, shape.answer,
				[]bedrock.Option{bedrock.WithModel(shape.model)},
				llms.WithRepetitionPenalty(1.1), llms.WithFrequencyPenalty(0.3),
				llms.WithPresencePenalty(0.7), llms.WithCandidateCount(3))

			got := bedrockWarningsByOption(resp.Warnings)
			for _, option := range shape.lost {
				require.Contains(t, got, option, "this payload has no field for it: %v", resp.Warnings)
			}
			for _, option := range shape.carried {
				require.NotContains(t, got, option, "this payload carries it: %v", resp.Warnings)
			}
		})
	}
}
