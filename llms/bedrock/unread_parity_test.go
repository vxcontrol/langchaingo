package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestBothBedrockDoorsReportTheOptionsNeitherReads(t *testing.T) {
	t.Parallel()

	for _, door := range []struct {
		name   string
		answer string
		opts   []bedrock.Option
	}{
		{"legacy", legacyAnswer, []bedrock.Option{
			bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0"),
		}},
		{"converse", converseAnswer, []bedrock.Option{
			bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0"), bedrock.WithConverseAPI(),
		}},
	} {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			resp := bedrockWarningsFor(t, door.answer, door.opts,
				llms.WithSeed(7), llms.WithVerbosity("low"),
				llms.WithMinLength(10), llms.WithMaxLength(20),
				llms.WithResponseMIMEType("application/json"))

			got := bedrockWarningsByOption(resp.Warnings)
			for _, option := range []string{
				"WithSeed", "WithVerbosity", "WithMinLength", "WithMaxLength", "WithResponseMIMEType",
			} {
				require.Contains(t, got, option, "the request has no field for it: %v", resp.Warnings)
			}
		})
	}
}
