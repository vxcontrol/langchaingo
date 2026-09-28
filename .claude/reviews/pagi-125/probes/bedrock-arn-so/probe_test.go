package bedrockclient

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/vxcontrol/langchaingo/llms"
)

type probeFakeRT struct{ got *bedrockruntime.ConverseInput }

func (f *probeFakeRT) Converse(_ context.Context, in *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	f.got = in
	return &bedrockruntime.ConverseOutput{
		StopReason: types.StopReasonEndTurn,
		Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Role:    types.ConversationRoleAssistant,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: `{"answer":"ok"}`}},
		}},
	}, nil
}

func (f *probeFakeRT) ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error) {
	return nil, fmt.Errorf("unused")
}

func TestProbeArnStructuredOutput(t *testing.T) {
	schema := `{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`
	for _, model := range []string{
		"arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4e5f6",
		"arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abcdef123456",
		"arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		"us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		"us.anthropic.claude-opus-4-7",
	} {
		f := &probeFakeRT{}
		_, err := NewConverseClient(f).CreateCompletionConverse(context.Background(), &ConverseInput{
			ModelID:          model,
			Messages:         []Message{{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"}},
			StructuredOutput: &llms.StructuredOutputConfig{Name: "a", Schema: json.RawMessage(schema)},
		})
		sent := f.got != nil && f.got.OutputConfig != nil
		t.Logf("model=%s err=%v outputConfigSent=%v", model, err, sent)
	}
}
