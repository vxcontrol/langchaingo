package bedrockclient

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"

	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnAnswerMadeOnlyOfReasoningIsStillAnAnswer(t *testing.T) {
	t.Parallel()

	block := &novaReasoningContentOutput{}
	block.ReasoningText.Text = "thinking"
	content, thought := splitNovaReasoning([]novaOutputContent{{ReasoningContent: block}})
	require.Empty(t, content)
	require.NotNil(t, thought, "the vendor did answer: with a thought and nothing else")
	assert.Equal(t, "thinking", thought.Content)
}

func TestAnEmptyAnswerIsStillAnError(t *testing.T) {
	t.Parallel()

	content, thought := splitNovaReasoning(nil)
	assert.Empty(t, content)
	assert.Nil(t, thought)
	var _ *reasoning.ContentReasoning = thought
}

func TestANovaTurnOfPureReasoningIsNotAnError(t *testing.T) {
	t.Parallel()

	client := bedrockruntime.New(bedrockruntime.Options{
		Region:           "us-east-1",
		Credentials:      credentials.NewStaticCredentialsProvider("k", "s", ""),
		RetryMaxAttempts: 1,
		HTTPClient: &cannedTransport{body: `{"output":{"message":{"content":[` +
			`{"reasoningContent":{"reasoningText":{"text":"thinking"}}}` +
			`],"role":"assistant"}},"stopReason":"end_turn",` +
			`"usage":{"inputTokens":1,"outputTokens":2}}`},
	})

	resp, err := createNovaCompletion(t.Context(), client, "amazon.nova-2-lite-v1:0",
		[]Message{{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"}},
		llms.CallOptions{}, &llms.Warnings{})
	require.NoError(t, err, "a thought with no text is still an answer, not a missing one")
	require.Len(t, resp.Choices, 1)
	require.NotNil(t, resp.Choices[0].Reasoning)
	assert.Equal(t, "thinking", resp.Choices[0].Reasoning.Content)
	assert.Equal(t, "end_turn", resp.Choices[0].StopReason)
}
