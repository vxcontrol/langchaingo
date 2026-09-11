package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestAskingForNoToolOnConverseIsNotReportedAsLosingTheTools(t *testing.T) {
	t.Parallel()

	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "get_weather", Description: "weather",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	}}

	resp := bedrockWarningsFor(t, converseAnswer,
		[]bedrock.Option{bedrock.WithModel("amazon.nova-lite-v1:0"), bedrock.WithConverseAPI()},
		llms.WithTools([]llms.Tool{tool}), llms.WithToolChoice("none"))

	require.NotContains(t, bedrockWarningsByOption(resp.Warnings), "WithTools",
		"the caller asked for no tool, so the empty config is the answer: %v", resp.Warnings)
}
