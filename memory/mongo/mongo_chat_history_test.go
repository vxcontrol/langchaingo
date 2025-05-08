package mongo

import (
	"context"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go/modules/mongodb"
)

func runTestContainer(t *testing.T) (string, error) {
	t.Helper()

	ctx := t.Context()

	mongoContainer, err := mongodb.Run(
		ctx,
		"mongo:7.0.8",
		mongodb.WithUsername("test"),
		mongodb.WithPassword("test"),
	)
	if err != nil {
		return "", err
	}

	t.Cleanup(func() {
		ctx := context.Background() //nolint:usetesting
		if err := mongoContainer.Terminate(ctx); err != nil {
			t.Fatalf("failed to terminate container: %s", err)
		}
	})

	url, err := mongoContainer.ConnectionString(ctx)
	if err != nil {
		return "", err
	}

	return url, nil
}

func TestMongoDBChatMessageHistory(t *testing.T) {
	t.Parallel()

	url, err := runTestContainer(t)
	require.NoError(t, err)

	ctx := t.Context()
	_, err = NewMongoDBChatMessageHistory(ctx, WithSessionID("test"))
	assert.Equal(t, errMongoInvalidURL, err)

	_, err = NewMongoDBChatMessageHistory(ctx, WithConnectionURL(url))
	assert.Equal(t, errMongoInvalidSessionID, err)

	history, err := NewMongoDBChatMessageHistory(ctx, WithConnectionURL(url), WithSessionID("testSessionXX"))
	require.NoError(t, err)

	err = history.AddAIMessage(ctx, "Hi")
	require.NoError(t, err)

	err = history.AddUserMessage(ctx, "Hello")
	require.NoError(t, err)

	messages, err := history.Messages(ctx)
	require.NoError(t, err)

	assert.Len(t, messages, 2)
	assert.Equal(t, llms.ChatMessageTypeAI, messages[0].GetType())
	assert.Equal(t, "Hi", messages[0].GetContent())
	assert.Equal(t, llms.ChatMessageTypeHuman, messages[1].GetType())
	assert.Equal(t, "Hello", messages[1].GetContent())
	t.Cleanup(func() {
		ctx := context.Background() //nolint:usetesting
		require.NoError(t, history.Clear(ctx))
	})
}
