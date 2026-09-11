// Package llmtest provides support for testing LLM implementations.
//
// Following the design of testing/fstest, this package provides a simple
// TestLLM function that verifies an LLM implementation behaves correctly.
package llmtest

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

// TestLLM tests an LLM implementation.
// It performs basic operations and checks that the model behaves correctly.
// It automatically discovers and tests capabilities by probing the model.
//
// If TestLLM finds any misbehaviors, it reports them via t.Error/t.Fatal.
//
// Typical usage inside a test:
//
//	func TestLLM(t *testing.T) {
//	    llm, err := mylllm.New(...)
//	    if err != nil {
//	        t.Fatal(err)
//	    }
//	    llmtest.TestLLM(t, llm)
//	}
func TestLLM(t *testing.T, model llms.Model, opts ...Option) {
	t.Helper()
	t.Parallel()

	var declared TestOptions
	for _, opt := range opts {
		opt(&declared)
	}

	// Run core tests as subtests - these should always work
	t.Run("Core", func(t *testing.T) {
		t.Parallel()

		t.Run("Call", func(t *testing.T) {
			t.Parallel()
			testCall(t, model)
		})

		t.Run("GenerateContent", func(t *testing.T) {
			t.Parallel()
			testGenerateContent(t, model)
		})
	})

	// Discover and test capabilities
	t.Run("Capabilities", func(t *testing.T) {
		t.Parallel()

		if !declared.SkipStreaming {
			t.Run("Streaming", func(t *testing.T) {
				t.Parallel()
				testStreaming(t, model)
			})
		}

		// Test tool calls if supported
		if supportsTools(model) {
			t.Run("ToolCalls", func(t *testing.T) {
				t.Parallel()
				testToolCalls(t, model)
			})
		}

		// Test caching by trying it - if it works, great
		t.Run("Caching", func(t *testing.T) {
			t.Parallel()
			testCaching(t, model)
		})

		// Test token counting - always run but don't fail if not supported
		t.Run("TokenCounting", func(t *testing.T) {
			t.Parallel()
			testTokenCounting(t, model)
		})
	})
}

// Capability detection functions

// supportsTools probes if the model supports tool calls
func supportsTools(model llms.Model) bool {
	// Try a simple tool call with a dummy tool
	ctx := context.Background()
	tools := []llms.Tool{
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "test_tool",
				Description: "Test tool",
				Parameters:  map[string]any{"type": "object"},
			},
		},
	}

	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("test"),
			},
		},
	}

	// Try with tools - if it doesn't error out, it's supported
	_, err := model.GenerateContent(ctx, messages,
		llms.WithTools(tools),
		llms.WithMaxTokens(1),
	)

	// If we get a specific "tools not supported" error, return false
	// Otherwise assume it's supported (even if other errors occur)
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "not support") {
		return false
	}
	return err == nil || !strings.Contains(strings.ToLower(err.Error()), "tool")
}

// TestLLMWithOptions tests an LLM with specific test options.
func TestLLMWithOptions(t *testing.T, model llms.Model, opts TestOptions, expected ...string) {
	t.Helper()

	// Store options for test functions to use
	testCtx := &testContext{
		model:    model,
		options:  opts,
		expected: expected,
	}

	// Run tests with context
	runTestsWithContext(t, testCtx)
}

// Option declares, at the call site, what the door under test cannot do.
type Option func(*TestOptions)

// WithoutStreaming declares a door that does not deliver a streaming callback,
// so the suite must not hold it to that contract.
func WithoutStreaming() Option {
	return func(o *TestOptions) { o.SkipStreaming = true }
}

// TestOptions configures test execution.
type TestOptions struct {
	// Timeout for each test operation
	Timeout time.Duration

	// Skip specific test categories
	SkipCall            bool
	SkipGenerateContent bool
	SkipStreaming       bool

	// Custom test prompts
	TestPrompt   string
	TestMessages []llms.MessageContent

	// For providers that need special options
	CallOptions []llms.CallOption
}

// Internal test context
type testContext struct {
	model    llms.Model
	options  TestOptions
	expected []string
}

func runTestsWithContext(t *testing.T, ctx *testContext) {
	behaviors := make(map[string]bool)
	for _, exp := range ctx.expected {
		behaviors[exp] = true
	}

	if !ctx.options.SkipCall {
		t.Run("Call", func(t *testing.T) {
			testCallWithContext(t, ctx)
		})
	}

	if !ctx.options.SkipGenerateContent {
		t.Run("GenerateContent", func(t *testing.T) {
			testGenerateContentWithContext(t, ctx)
		})
	}

	if behaviors["supports-streaming"] && !ctx.options.SkipStreaming {
		t.Run("Streaming", func(t *testing.T) {
			testStreamingWithContext(t, ctx)
		})
	}
}

// Core test implementations

func testCall(t *testing.T, model llms.Model) {
	t.Helper()
	ctx := context.Background()

	result, err := llms.GenerateFromSinglePrompt(ctx, model, "Reply with 'OK' and nothing else", llms.WithMaxTokens(10))
	if err != nil {
		t.Fatalf("Call failed: %v", err)
	}

	if result == "" {
		t.Error("Call returned empty result")
	}
}

func testCallWithContext(t *testing.T, tctx *testContext) {
	t.Helper()
	ctx := context.Background()
	if tctx.options.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, tctx.options.Timeout)
		defer cancel()
	}

	prompt := "Reply with 'OK' and nothing else"
	if tctx.options.TestPrompt != "" {
		prompt = tctx.options.TestPrompt
	}

	opts := append([]llms.CallOption{llms.WithMaxTokens(10)}, tctx.options.CallOptions...)
	result, err := llms.GenerateFromSinglePrompt(ctx, tctx.model, prompt, opts...)
	if err != nil {
		t.Fatalf("Call failed: %v", err)
	}

	if result == "" {
		t.Error("Call returned empty result")
	}
}

func testGenerateContent(t *testing.T, model llms.Model) {
	t.Helper()
	ctx := context.Background()

	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Reply with 'Hello' and nothing else"),
			},
		},
	}

	resp, err := model.GenerateContent(ctx, messages, llms.WithMaxTokens(10))
	if err != nil {
		t.Fatalf("GenerateContent failed: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("No choices in response")
	}

	content := strings.ToLower(resp.Choices[0].Content)
	if !strings.Contains(content, "hello") {
		t.Errorf("Expected 'hello' in response, got: %s", resp.Choices[0].Content)
	}
}

func testGenerateContentWithContext(t *testing.T, tctx *testContext) {
	t.Helper()
	ctx := context.Background()
	if tctx.options.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, tctx.options.Timeout)
		defer cancel()
	}

	messages := tctx.options.TestMessages
	if len(messages) == 0 {
		messages = []llms.MessageContent{
			{
				Role: llms.ChatMessageTypeHuman,
				Parts: []llms.ContentPart{
					llms.TextPart("Reply with 'Hello' and nothing else"),
				},
			},
		}
	}

	opts := append([]llms.CallOption{llms.WithMaxTokens(10)}, tctx.options.CallOptions...)
	resp, err := tctx.model.GenerateContent(ctx, messages, opts...)
	if err != nil {
		t.Fatalf("GenerateContent failed: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("No choices in response")
	}
}

func testStreaming(t *testing.T, model llms.Model) {
	t.Helper()

	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Count from 1 to 3"),
			},
		},
	}

	assertStreams(t, context.Background(), model, messages, llms.WithMaxTokens(50))
}

func testStreamingWithContext(t *testing.T, tctx *testContext) {
	t.Helper()
	ctx := context.Background()
	if tctx.options.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, tctx.options.Timeout)
		defer cancel()
	}

	messages := tctx.options.TestMessages
	if len(messages) == 0 {
		messages = []llms.MessageContent{
			{
				Role: llms.ChatMessageTypeHuman,
				Parts: []llms.ContentPart{
					llms.TextPart("Count from 1 to 3"),
				},
			},
		}
	}

	opts := append([]llms.CallOption{llms.WithMaxTokens(50)}, tctx.options.CallOptions...)
	assertStreams(t, ctx, tctx.model, messages, opts...)
}

func assertStreams(t *testing.T, ctx context.Context, model llms.Model, messages []llms.MessageContent, opts ...llms.CallOption) {
	t.Helper()

	var mu sync.Mutex
	var streamed strings.Builder
	collect := llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
		if chunk.Type != streaming.ChunkTypeText {
			return nil
		}
		mu.Lock()
		defer mu.Unlock()
		streamed.WriteString(chunk.Content)
		return nil
	})

	resp, err := model.GenerateContent(ctx, messages, append(opts, collect)...)
	if err != nil {
		t.Fatalf("GenerateContent with a streaming callback failed: %v", err)
	}

	mu.Lock()
	got := streamed.String()
	mu.Unlock()

	if got == "" {
		t.Fatal("the door took a streaming callback and never called it with text")
	}
	if len(resp.Choices) == 0 {
		t.Fatal("streaming call returned no choices")
	}
	if content := resp.Choices[0].Content; content != got {
		t.Errorf("the streamed text and the returned content disagree:\n streamed: %q\n returned: %q", got, content)
	}
}

func testToolCalls(t *testing.T, model llms.Model) {
	t.Helper()
	ctx := context.Background()

	// Define a simple tool
	tools := []llms.Tool{
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city and country",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("What's the weather in San Francisco?"),
			},
		},
	}

	resp, err := model.GenerateContent(ctx, messages,
		llms.WithTools(tools),
		llms.WithMaxTokens(100),
	)
	if err != nil {
		t.Fatalf("GenerateContent with tools failed: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("No choices in response")
	}

	// Check if tool was called
	choice := resp.Choices[0]
	if len(choice.ToolCalls) == 0 {
		t.Log("No tool calls in response (model may not support tools)")
	} else {
		toolCall := choice.ToolCalls[0]
		if toolCall.FunctionCall.Name != "get_weather" {
			t.Errorf("Expected get_weather tool call, got: %s", toolCall.FunctionCall.Name)
		}
	}
}

func testCaching(t *testing.T, model llms.Model) {
	t.Helper()
	ctx := context.Background()

	// Long context that benefits from caching
	longContext := strings.Repeat("This is cached context. ", 50)

	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeSystem,
			Parts: []llms.ContentPart{
				llms.TextPart(longContext),
			},
		},
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Say 'OK'"),
			},
		},
	}

	// First call (cache miss)
	_, err := model.GenerateContent(ctx, messages, llms.WithMaxTokens(10))
	if err != nil {
		t.Fatalf("First call failed: %v", err)
	}

	// Second call (potential cache hit)
	resp2, err := model.GenerateContent(ctx, messages, llms.WithMaxTokens(10))
	if err != nil {
		t.Fatalf("Second call failed: %v", err)
	}

	// Check if caching info is available
	if genInfo := resp2.Choices[0].GenerationInfo; genInfo != nil {
		if cached, ok := genInfo["PromptCachedTokens"].(int); ok && cached > 0 {
			t.Logf("Cached %d tokens", cached)
		}
	}
}

func testTokenCounting(t *testing.T, model llms.Model) {
	t.Helper()
	ctx := context.Background()

	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("Count to 5"),
			},
		},
	}

	resp, err := model.GenerateContent(ctx, messages, llms.WithMaxTokens(50))
	if err != nil {
		t.Fatalf("GenerateContent failed: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("No choices in response")
	}

	genInfo := resp.Choices[0].GenerationInfo
	if genInfo == nil {
		t.Skip("No generation info provided")
	}

	// Check for token counts
	var hasTokenInfo bool
	for _, field := range []string{"TotalTokens", "PromptTokens", "CompletionTokens"} {
		if v, ok := genInfo[field].(int); ok && v > 0 {
			hasTokenInfo = true
			t.Logf("%s: %d", field, v)
		}
	}

	if !hasTokenInfo {
		t.Log("No token counting information provided")
	}
}

// ValidateLLM checks if a model satisfies basic requirements without running tests.
// It returns an error describing what's wrong, or nil if the model is valid.
func ValidateLLM(model llms.Model) error {
	if model == nil {
		return errors.New("model is nil")
	}

	// Check if required methods are implemented
	ctx := context.Background()

	// Try a simple call
	_, err := llms.GenerateFromSinglePrompt(ctx, model, "test", llms.WithMaxTokens(1))
	if err != nil {
		return fmt.Errorf("Call method failed: %w", err)
	}

	// Try GenerateContent
	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextPart("test"),
			},
		},
	}
	_, err = model.GenerateContent(ctx, messages, llms.WithMaxTokens(1))
	if err != nil {
		return fmt.Errorf("GenerateContent method failed: %w", err)
	}

	return nil
}

// MockLLM provides a simple mock implementation for testing.
type MockLLM struct {
	// Response to return from Call
	CallResponse string
	CallError    error

	// Response to return from GenerateContent
	GenerateResponse *llms.ContentResponse
	GenerateError    error

	// Track calls for verification
	mu            sync.Mutex
	CallCount     int
	GenerateCount int
	LastPrompt    string
	LastMessages  []llms.MessageContent
}

// Call implements llms.Model
func (m *MockLLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	m.mu.Lock()
	m.CallCount++
	m.LastPrompt = prompt
	m.mu.Unlock()
	return m.CallResponse, m.CallError
}

// GenerateContent implements llms.Model
func (m *MockLLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	m.mu.Lock()
	m.GenerateCount++
	m.LastMessages = messages
	m.mu.Unlock()

	response := m.GenerateResponse
	if response == nil {
		response = &llms.ContentResponse{
			Choices: []*llms.ContentChoice{
				{
					Content: "mock response",
				},
			},
		}
	}

	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}
	if opts.StreamingFunc != nil && len(response.Choices) > 0 {
		var sent strings.Builder
		for _, part := range strings.SplitAfter(response.Choices[0].Content, " ") {
			sent.WriteString(part)
			if err := streaming.CallWithText(ctx, opts.StreamingFunc, part); err != nil {
				return partialMockResponse(sent.String()), err
			}
		}
		if err := streaming.CallWithDone(ctx, opts.StreamingFunc); err != nil {
			return partialMockResponse(sent.String()), err
		}
	}

	return response, m.GenerateError
}

func partialMockResponse(content string) *llms.ContentResponse {
	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{Content: content}},
	}
}

// Verify MockLLM implements llms.Model
var _ llms.Model = (*MockLLM)(nil)
