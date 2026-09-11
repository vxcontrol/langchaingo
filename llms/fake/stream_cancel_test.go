package fake_test

import (
	"context"
	"errors"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/fake"
	"github.com/vxcontrol/langchaingo/llms/streaming"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var errGaveUp = errors.New("the consumer gave up")

func TestAConsumerThatGivesUpStillGetsWhatArrived(t *testing.T) {
	t.Parallel()

	llm := fake.NewFakeLLM([]string{"sixty rooms are free"})

	delivered := 0
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			if chunk.Type != streaming.ChunkTypeText {
				return nil
			}
			delivered++
			if delivered == 2 {
				return errGaveUp
			}
			return nil
		}))

	require.ErrorIs(t, err, errGaveUp)
	require.NotNil(t, resp, "a real door hands back the text it collected; the fake must too")
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "sixty rooms ", resp.Choices[0].Content)
}

func TestAStreamNobodyInterruptsStillReturnsEverything(t *testing.T) {
	t.Parallel()

	llm := fake.NewFakeLLM([]string{"sixty rooms are free"})

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	require.NoError(t, err)
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
}
