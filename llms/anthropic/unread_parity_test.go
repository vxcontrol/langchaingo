package anthropic_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestTheAnthropicDoorReportsTheOptionsItNeverReads(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithSeed(7), llms.WithVerbosity("low"),
		llms.WithMinLength(10), llms.WithMaxLength(20),
		llms.WithResponseMIMEType("application/json"))

	got := warningsByOption(resp.Warnings)
	for _, option := range []string{
		"WithSeed", "WithVerbosity", "WithMinLength", "WithMaxLength", "WithResponseMIMEType",
	} {
		require.Contains(t, got, option, "the request has no field for it: %v", resp.Warnings)
	}
}
