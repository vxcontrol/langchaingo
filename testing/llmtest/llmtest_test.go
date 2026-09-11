package llmtest

import (
	"context"
	"errors"
	"os"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMockLLM tests the mock implementation.
func TestMockLLM(t *testing.T) {
	mock := &MockLLM{
		CallResponse: "OK",
		GenerateResponse: &llms.ContentResponse{
			Choices: []*llms.ContentChoice{
				{
					Content: "Hello",
					GenerationInfo: map[string]interface{}{
						"TotalTokens": 10,
					},
				},
			},
		},
	}

	TestLLM(t, mock)
}

// TestValidateLLM tests the validation function.
func TestValidateLLM(t *testing.T) {
	// Test with nil model
	if err := ValidateLLM(nil); err == nil {
		t.Error("ValidateLLM should fail with nil model")
	}

	// Test with valid mock
	mock := &MockLLM{
		CallResponse: "OK",
		GenerateResponse: &llms.ContentResponse{
			Choices: []*llms.ContentChoice{
				{
					Content: "response",
				},
			},
		},
	}

	if err := ValidateLLM(mock); err != nil {
		t.Errorf("ValidateLLM failed with valid mock: %v", err)
	}
}

// Integration tests with real providers (require API keys)

func TestAnthropicIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test")
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	// Import is handled in the actual test files for each provider
}

func TestOpenAIIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test")
	}

	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	// Import is handled in the actual test files for each provider
}

func TestTheMockHandsBackWhatTheConsumerReceivedBeforeGivingUp(t *testing.T) {
	t.Parallel()

	mock := &MockLLM{GenerateResponse: &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{Content: "sixty rooms are free"}},
	}}

	gaveUp := errors.New("the consumer gave up")
	delivered := 0
	resp, err := mock.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			if chunk.Type != streaming.ChunkTypeText {
				return nil
			}
			delivered++
			if delivered == 2 {
				return gaveUp
			}
			return nil
		}))

	require.ErrorIs(t, err, gaveUp)
	require.NotNil(t, resp, "a real door hands back the text it collected; the mock must too")
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "sixty rooms ", resp.Choices[0].Content)
}
