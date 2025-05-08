package memory

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatMessageHistory(t *testing.T) {
	t.Parallel()

	h := NewChatMessageHistory()
	err := h.AddAIMessage(t.Context(), "foo")
	require.NoError(t, err)
	err = h.AddUserMessage(t.Context(), "bar")
	require.NoError(t, err)

	messages, err := h.Messages(t.Context())
	require.NoError(t, err)

	assert.Equal(t, []llms.ChatMessage{
		llms.AIChatMessage{Content: "foo"},
		llms.HumanChatMessage{Content: "bar"},
	}, messages)

	h = NewChatMessageHistory(
		WithPreviousMessages([]llms.ChatMessage{
			llms.AIChatMessage{Content: "foo"},
			llms.SystemChatMessage{Content: "bar"},
		}),
	)
	err = h.AddUserMessage(t.Context(), "zoo")
	require.NoError(t, err)

	messages, err = h.Messages(t.Context())
	require.NoError(t, err)

	assert.Equal(t, []llms.ChatMessage{
		llms.AIChatMessage{Content: "foo"},
		llms.SystemChatMessage{Content: "bar"},
		llms.HumanChatMessage{Content: "zoo"},
	}, messages)
}
