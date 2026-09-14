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

func TestConverseReportsATopKThatNeverReachesTheRequest(t *testing.T) {
	t.Parallel()

	topK := 40
	resp := converseCall(t, &ConverseInput{
		Messages:        humanTurn(),
		ModelID:         "us.anthropic.claude-sonnet-4-5-v1:0",
		TopK:            &topK,
		ReasoningConfig: &llms.ReasoningConfig{Mode: llms.ReasoningOn, Effort: llms.ReasoningMedium},
	})

	w, ok := converseWarningsByOption(resp.Warnings)["WithTopK"]
	require.True(t, ok, "no top-k warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "40", w.Asked)
}

func TestConverseStaysSilentOnATopKItCarries(t *testing.T) {
	t.Parallel()

	topK := 40
	resp := converseCall(t, &ConverseInput{
		Messages: humanTurn(),
		ModelID:  "us.anthropic.claude-sonnet-4-5-v1:0",
		TopK:     &topK,
	})

	require.NotContains(t, converseWarningsByOption(resp.Warnings), "WithTopK")
}

func TestConverseReportsAThinkingBudgetItCut(t *testing.T) {
	t.Parallel()

	maxTokens := 4096
	resp := converseCall(t, &ConverseInput{
		Messages:        humanTurn(),
		ModelID:         "us.anthropic.claude-sonnet-4-5-v1:0",
		MaxTokens:       &maxTokens,
		ReasoningConfig: &llms.ReasoningConfig{Mode: llms.ReasoningOn, Tokens: 30000},
	})

	w, ok := converseWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, w.Kind)
	require.Equal(t, "30000 tokens", w.Asked)
	require.NotEqual(t, w.Asked, w.Sent)
}

func TestConverseReportsAnEffortItLowered(t *testing.T) {
	t.Parallel()

	maxTokens := 8000
	resp := converseCall(t, &ConverseInput{
		Messages:        humanTurn(),
		ModelID:         "anthropic.claude-opus-4-6-v1:0",
		MaxTokens:       &maxTokens,
		ReasoningConfig: &llms.ReasoningConfig{Mode: llms.ReasoningOn, Effort: llms.ReasoningXHigh},
	})

	w, ok := converseWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, w.Kind)
	require.Equal(t, "xhigh", w.Asked)
	require.Equal(t, "high", w.Sent)
}

func TestConverseReportsAToolChoiceItTurnsIntoAuto(t *testing.T) {
	t.Parallel()

	resp := converseCall(t, &ConverseInput{
		Messages: []Message{
			{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"},
			{Role: llms.ChatMessageTypeAI, Type: "tool_use", ToolCall: &ToolCall{
				ID: "t1", Name: "get_weather", Arguments: map[string]any{},
			}},
			{Role: llms.ChatMessageTypeTool, Type: "tool_result", ToolResult: &ToolResult{
				ToolCallID: "t1", ToolName: "get_weather", Content: "sunny",
			}},
		},
		ModelID:    "anthropic.claude-sonnet-4-5-v1:0",
		Tools:      []llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{Name: "get_weather"}}},
		ToolChoice: "none",
	})

	w, ok := converseWarningsByOption(resp.Warnings)["WithToolChoice"]
	require.True(t, ok, "no tool-choice warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningSubstitute, w.Kind)
	require.Equal(t, "none", w.Asked)
	require.Equal(t, "auto", w.Sent)
}

func TestConverseReadsTheEffortEveryFamilyWritesItsOwnWay(t *testing.T) {
	t.Parallel()

	maxTokens := 8192
	for _, tc := range []struct {
		model  string
		effort llms.ReasoningEffort
		want   *llms.Warning
	}{
		{"us.amazon.nova-2-lite-v1:0", llms.ReasoningHigh, nil},
		{"us.xai.grok-4.3", llms.ReasoningMax, &llms.Warning{
			Kind: llms.WarningClamp, Option: "WithReasoning", Asked: "max", Sent: "xhigh",
		}},
		{"openai.gpt-oss-120b-1:0", llms.ReasoningLow, nil},
		{"openai.gpt-oss-20b-1:0", llms.ReasoningMax, &llms.Warning{
			Kind: llms.WarningClamp, Option: "WithReasoning", Asked: "max", Sent: "high",
		}},
	} {
		t.Run(tc.model, func(t *testing.T) {
			t.Parallel()

			resp := converseCall(t, &ConverseInput{
				Messages:        humanTurn(),
				ModelID:         tc.model,
				MaxTokens:       &maxTokens,
				ReasoningConfig: &llms.ReasoningConfig{Mode: llms.ReasoningOn, Effort: tc.effort},
			})

			got, ok := converseWarningsByOption(resp.Warnings)["WithReasoning"]
			if tc.want == nil {
				require.False(t, ok, "the effort reached the wire, so nothing was lost: %v", resp.Warnings)
				return
			}
			require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
			require.Equal(t, tc.want.Kind, got.Kind)
			require.Equal(t, tc.want.Asked, got.Asked)
			require.Equal(t, tc.want.Sent, got.Sent)
		})
	}
}

func TestConverseReportsAMechanismTheModelDoesNotOffer(t *testing.T) {
	t.Parallel()

	maxTokens := 8192
	resp := converseCall(t, &ConverseInput{
		Messages:  humanTurn(),
		ModelID:   "us.anthropic.claude-sonnet-4-5-v1:0",
		MaxTokens: &maxTokens,
		ReasoningConfig: &llms.ReasoningConfig{
			Mode: llms.ReasoningOn, Effort: llms.ReasoningHigh, Adaptive: true,
		},
	})

	w, ok := converseWarningsByOption(resp.Warnings)["WithAdaptiveReasoning"]
	require.True(t, ok, "no mechanism warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningSubstitute, w.Kind)
	require.Equal(t, "adaptive", w.Asked)
	require.Equal(t, "enabled", w.Sent)
}
