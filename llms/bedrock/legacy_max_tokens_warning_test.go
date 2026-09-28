package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestAnAnswerLimitTheLegacyDoorNamedItselfIsReported(t *testing.T) {
	t.Parallel()

	const cohereAnswer = `{"generations":[{"text":"ok","finish_reason":"COMPLETE"}]}`

	for _, payload := range []struct {
		name   string
		model  string
		answer string
		sent   string
	}{
		{"anthropic", "anthropic.claude-sonnet-4-5-v1:0", legacyAnswer, "2048"},
		{"cohere", "cohere.command-text-v14", cohereAnswer, "20"},
	} {
		t.Run(payload.name, func(t *testing.T) {
			t.Parallel()

			resp := bedrockWarningsFor(t, payload.answer,
				[]bedrock.Option{bedrock.WithModel(payload.model)},
				llms.WithMaxTokens(0))

			w, ok := bedrockWarningsByOption(resp.Warnings)["WithMaxTokens"]
			require.True(t, ok, "the door named a limit the caller did not: %v", resp.Warnings)
			require.Equal(t, llms.WarningSubstitute, w.Kind)
			require.Equal(t, "0", w.Asked)
			require.Equal(t, payload.sent, w.Sent)
		})
	}
}

func TestALimitTheLegacyDoorCarriesAsAskedIsNotReported(t *testing.T) {
	t.Parallel()

	resp := bedrockWarningsFor(t, legacyAnswer,
		[]bedrock.Option{bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0")},
		llms.WithMaxTokens(1024))

	require.NotContains(t, bedrockWarningsByOption(resp.Warnings), "WithMaxTokens",
		"the limit travelled as asked: %v", resp.Warnings)
}
