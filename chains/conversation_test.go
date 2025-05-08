package chains

import (
	"context"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/memory"
	"github.com/vxcontrol/langchaingo/memory/zep"

	z "github.com/getzep/zep-go"
	zClient "github.com/getzep/zep-go/client"
	zOption "github.com/getzep/zep-go/option"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func TestConversation(t *testing.T) {
	t.Parallel()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	llm, err := openai.New()
	require.NoError(t, err)

	c := NewConversation(llm, memory.NewConversationBuffer())
	_, err = Run(t.Context(), c, "Hi! I'm Jim")
	require.NoError(t, err)

	res, err := Run(t.Context(), c, "What is my name?")
	require.NoError(t, err)
	require.True(t, strings.Contains(res, "Jim"), `result does not contain the keyword 'Jim'`)
}

func TestConversationWithZepMemory(t *testing.T) {
	t.Parallel()

	if zepKey := os.Getenv("ZEP_API_KEY"); zepKey == "" {
		t.Skip("ZEP_API_KEY not set")
	}
	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	llm, err := openai.New()
	require.NoError(t, err)

	zc := zClient.NewClient(
		zOption.WithAPIKey(os.Getenv("ZEP_API_KEY")),
	)
	sessionID := os.Getenv("ZEP_SESSION_ID")
	if sessionID == "" {
		sessionID = setupZepSession(t.Context(), t, zc)
	}

	c := NewConversation(
		llm,
		zep.NewMemory(
			zc,
			sessionID,
			zep.WithMemoryType(z.MemoryGetRequestMemoryTypePerpetual),
			zep.WithHumanPrefix("Joe"),
			zep.WithAIPrefix("Robot"),
		),
	)
	_, err = Run(t.Context(), c, "Hi! I'm Jim")
	require.NoError(t, err)

	res, err := Run(t.Context(), c, "What is my name?")
	require.NoError(t, err)
	require.True(t, strings.Contains(res, "Jim"), `result does not contain the keyword 'Jim'`)
}

func TestConversationWithChatLLM(t *testing.T) {
	t.Parallel()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	llm, err := openai.New()
	require.NoError(t, err)

	c := NewConversation(llm, memory.NewConversationTokenBuffer(llm, 2000))
	_, err = Run(t.Context(), c, "Hi! I'm Jim")
	require.NoError(t, err)

	res, err := Run(t.Context(), c, "What is my name?")
	require.NoError(t, err)
	require.True(t, strings.Contains(res, "Jim"), `result does contain the keyword 'Jim'`)

	// this message will hit the maxTokenLimit and will initiate the prune of the messages to fit the context
	res, err = Run(t.Context(), c, "Are you sure that my name is Jim?")
	require.NoError(t, err)
	require.True(t, strings.Contains(res, "Jim"), `result does contain the keyword 'Jim'`)
}

func setupZepSession(ctx context.Context, t *testing.T, zc *zClient.Client) string {
	t.Helper()

	var (
		user    *z.User
		session *z.Session
	)

	firstName, lastName := "Langchaingo", "Test"
	users, err := zc.User.ListOrdered(ctx, &z.UserListOrderedRequest{})
	require.NoError(t, err)
	if len(users.Users) > 0 {
		idx := slices.IndexFunc(users.Users, func(u *z.User) bool {
			return u.FirstName != nil && *u.FirstName == firstName && u.LastName != nil && *u.LastName == lastName
		})
		require.NotEqual(t, -1, idx, "user not found")
		user = users.Users[idx]
	} else {
		userID := "langchaingo-test"
		email := "langchaingo@example.com"

		user, err = zc.User.Add(ctx, &z.CreateUserRequest{
			UserID:    &userID,
			Email:     &email,
			FirstName: &firstName,
			LastName:  &lastName,
		})
		require.NoError(t, err)
	}

	sessionID := uuid.New().String()
	session, err = zc.Memory.AddSession(ctx, &z.CreateSessionRequest{
		SessionID: sessionID,
		UserID:    user.UserID,
	})
	require.NoError(t, err)

	return *session.SessionID
}
