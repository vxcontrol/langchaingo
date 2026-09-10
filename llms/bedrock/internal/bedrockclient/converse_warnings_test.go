package bedrockclient

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func converseCall(t *testing.T, in *ConverseInput) *llms.ContentResponse {
	t.Helper()

	mockClient := &MockBedrockRuntimeClient{}
	mockClient.On("Converse", mock.Anything, mock.Anything, mock.Anything).Return(
		&bedrockruntime.ConverseOutput{
			Output: &types.ConverseOutputMemberMessage{
				Value: types.Message{
					Role:    types.ConversationRoleAssistant,
					Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "hi"}},
				},
			},
			StopReason: types.StopReasonEndTurn,
		}, nil)

	resp, err := NewConverseClient(mockClient).CreateCompletionConverse(context.Background(), in)
	require.NoError(t, err)
	return resp
}

func converseWarningsByOption(warnings []llms.Warning) map[string]llms.Warning {
	byOption := make(map[string]llms.Warning, len(warnings))
	for _, w := range warnings {
		byOption[w.Option] = w
	}
	return byOption
}

func TestConverseReportsTheSamplingBudgetThinkingReshapes(t *testing.T) {
	t.Parallel()

	temperature, topP, maxTokens := 0.2, 0.9, 1000
	resp := converseCall(t, &ConverseInput{
		Messages:        humanTurn(),
		ModelID:         "us.anthropic.claude-sonnet-4-5-v1:0",
		Temperature:     &temperature,
		TopP:            &topP,
		MaxTokens:       &maxTokens,
		ReasoningConfig: &llms.ReasoningConfig{Mode: llms.ReasoningOn, Effort: llms.ReasoningMedium},
	})

	got := converseWarningsByOption(resp.Warnings)

	temp, ok := got["WithTemperature"]
	require.True(t, ok, "no temperature warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningSubstitute, temp.Kind)
	require.Equal(t, "0.2", temp.Asked)
	require.Equal(t, "1", temp.Sent)

	tp, ok := got["WithTopP"]
	require.True(t, ok, "no top-p warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, tp.Kind)
	require.Empty(t, tp.Sent)

	limit, ok := got["WithMaxTokens"]
	require.True(t, ok, "no max-tokens warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, limit.Kind)
	require.Equal(t, "1000", limit.Asked)
	require.NotEqual(t, limit.Asked, limit.Sent)
}

func TestConverseReportsTopPDroppedForTemperature(t *testing.T) {
	t.Parallel()

	temperature, topP, maxTokens := 0.2, 0.9, 1000
	resp := converseCall(t, &ConverseInput{
		Messages:    humanTurn(),
		ModelID:     "us.anthropic.claude-sonnet-4-5-v1:0",
		Temperature: &temperature,
		TopP:        &topP,
		MaxTokens:   &maxTokens,
	})

	got := converseWarningsByOption(resp.Warnings)
	tp, ok := got["WithTopP"]
	require.True(t, ok, "no top-p warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, tp.Kind)
	require.Equal(t, "0.9", tp.Asked)
	require.NotContains(t, got, "WithTemperature", "the temperature this model keeps is not a loss")
}

func TestConverseCarriesAPlainRequestWithoutWarnings(t *testing.T) {
	t.Parallel()

	temperature, maxTokens := 0.2, 1000
	resp := converseCall(t, &ConverseInput{
		Messages:    humanTurn(),
		ModelID:     "us.anthropic.claude-sonnet-4-5-v1:0",
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
	})

	require.Empty(t, resp.Warnings)
}
