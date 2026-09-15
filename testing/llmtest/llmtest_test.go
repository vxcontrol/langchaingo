package llmtest

import (
	"context"
	"errors"
	"os"
	"runtime"
	"sync/atomic"
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

	TestLLM(t, mock, WithoutToolCalls())
}

type verdictRecorder struct {
	testing.TB
	failed atomic.Bool
}

func (r *verdictRecorder) Fail()                 { r.failed.Store(true) }
func (r *verdictRecorder) Error(...any)          { r.Fail() }
func (r *verdictRecorder) Errorf(string, ...any) { r.Fail() }
func (r *verdictRecorder) FailNow()              { r.Fail(); runtime.Goexit() }
func (r *verdictRecorder) Fatal(...any)          { r.FailNow() }
func (r *verdictRecorder) Fatalf(string, ...any) { r.FailNow() }
func (r *verdictRecorder) SkipNow()              { runtime.Goexit() }
func (r *verdictRecorder) Skip(...any)           { r.SkipNow() }
func (r *verdictRecorder) Skipf(string, ...any)  { r.SkipNow() }

func failsToolCalls(t *testing.T, model llms.Model) bool {
	t.Helper()

	recorder := &verdictRecorder{TB: t}
	done := make(chan struct{})
	go func() {
		defer close(done)
		testToolCalls(recorder, model)
	}()
	<-done
	return recorder.failed.Load()
}

func TestTheSuiteFailsADoorThatTakesToolsAndCallsNone(t *testing.T) {
	t.Parallel()

	prose := &MockLLM{GenerateResponse: &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{Content: "it is sunny in San Francisco"}},
	}}

	assert.True(t, failsToolCalls(t, prose),
		"a door that answers in prose when the question needs the offered tool breaks the tool contract")
}

func TestTheSuitePassesADoorThatCallsTheTool(t *testing.T) {
	t.Parallel()

	caller := &MockLLM{GenerateResponse: &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{ToolCalls: []llms.ToolCall{{
			ID: "call_1", Type: "function",
			FunctionCall: &llms.FunctionCall{Name: "get_weather", Arguments: `{"location":"San Francisco, US"}`},
		}}}},
	}}

	assert.False(t, failsToolCalls(t, caller))
}

func TestTheSuiteSparesADoorThatReportsTheToolsDropped(t *testing.T) {
	dropping := &MockLLM{GenerateResponse: &llms.ContentResponse{
		Choices:  []*llms.ContentChoice{{Content: "Hello"}},
		Warnings: []llms.Warning{{Kind: llms.WarningDrop, Option: "WithTools", Asked: "1 tools"}},
	}}

	TestLLM(t, dropping)
}

func TestTheToolProbeHoldsADoorThatDoesNotReportTheToolsDropped(t *testing.T) {
	t.Parallel()

	for name, warnings := range map[string][]llms.Warning{
		"no warning":             nil,
		"another option dropped": {{Kind: llms.WarningDrop, Option: "WithSeed", Asked: "7"}},
		"the tools clamped":      {{Kind: llms.WarningClamp, Option: "WithTools", Asked: "2 tools", Sent: "1 tools"}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			door := &MockLLM{GenerateResponse: &llms.ContentResponse{
				Choices: []*llms.ContentChoice{{Content: "Hello"}}, Warnings: warnings,
			}}
			assert.True(t, supportsTools(door))
		})
	}
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
