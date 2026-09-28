package agents_test

import (
	"net/http"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/agents"
	"github.com/vxcontrol/langchaingo/chains"
	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/prompts"
	"github.com/vxcontrol/langchaingo/tools"

	"github.com/stretchr/testify/require"
)

func TestOpenAIFunctionsAgentWithHTTPRR(t *testing.T) {
	ctx := t.Context()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	if !rr.Recording() {
		t.Parallel()
	}

	// Configure OpenAI client with httprr
	opts := []openai.Option{
		openai.WithModel("gpt-4o"),
		openai.WithHTTPClient(rr.Client()),
	}
	if rr.Replaying() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}

	llm, err := openai.New(opts...)
	if err != nil {
		t.Fatal(err)
	}

	// Create a simple calculator tool
	calculator := tools.Calculator{}

	// Create the OpenAI Functions agent
	agent := agents.NewOpenAIFunctionsAgent(
		llm,
		[]tools.Tool{calculator},
		agents.NewOpenAIOption().WithSystemMessage("You are a helpful assistant that can perform calculations."),
	)

	// Create executor
	executor := agents.NewExecutor(agent)

	// Run a simple calculation
	result, err := chains.Run(ctx, executor, "What is 15 multiplied by 4?")
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Agent response: %s", result)

	// Verify the result contains 60
	if !strings.Contains(result, "60") {
		t.Errorf("expected calculation result 60 in response, got: %s", result)
	}
}

func TestOpenAIFunctionsAgentComplexCalculation(t *testing.T) {
	ctx := t.Context()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	if !rr.Recording() {
		t.Parallel()
	}

	// Configure OpenAI client with httprr
	opts := []openai.Option{
		openai.WithModel("gpt-4o"),
		openai.WithHTTPClient(rr.Client()),
	}
	if rr.Replaying() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)

	// Create a calculator tool
	calculator := tools.Calculator{}

	// Create the OpenAI Functions agent with extra messages
	agent := agents.NewOpenAIFunctionsAgent(
		llm,
		[]tools.Tool{calculator},
		agents.NewOpenAIOption().WithSystemMessage("You are a helpful math assistant."),
		agents.NewOpenAIOption().WithExtraMessages([]prompts.MessageFormatter{
			prompts.NewHumanMessagePromptTemplate("Please show your work step by step.", nil),
		}),
	)

	// Create executor with options
	executor := agents.NewExecutor(
		agent,
		agents.WithMaxIterations(5),
	)

	// Run a more complex calculation
	result, err := chains.Run(ctx, executor, "If I have 3 groups of 7 items, and I add 9 more items, how many items do I have in total?")
	if err != nil {
		t.Fatalf("failed to run agent: %v", err)
	}
	t.Logf("Agent response: %s", result)

	// Verify the result contains 30 (3*7 + 9 = 21 + 9 = 30)
	if !strings.Contains(result, "30") {
		t.Errorf("expected calculation result 30 in response, got: %s", result)
	}
}

// TestOpenAIFunctionsAgent_ParseOutput_NilResponse tests that ParseOutput handles nil response gracefully
func TestOpenAIFunctionsAgent_ParseOutput_NilResponse(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Test with nil response - should return error instead of panic
	_, _, err := agent.ParseOutput(nil)
	if err == nil {
		t.Error("expected error for nil response")
	}
}

// TestOpenAIFunctionsAgent_ParseOutput_EmptyChoices tests that ParseOutput handles empty choices gracefully
func TestOpenAIFunctionsAgent_ParseOutput_EmptyChoices(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Test with empty choices - should return error instead of panic
	resp := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{},
	}
	_, _, err := agent.ParseOutput(resp)
	if err == nil {
		t.Error("expected error for empty choices")
	}
}

// TestOpenAIFunctionsAgent_ParseOutput_MultipleToolCalls tests multiple tool calls handling
func TestOpenAIFunctionsAgent_ParseOutput_MultipleToolCalls(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Test multiple tool calls - should handle all calls, not just first one
	resp := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				ToolCalls: []llms.ToolCall{
					{
						ID: "call1",
						FunctionCall: &llms.FunctionCall{
							Name:      "calculator",
							Arguments: `{"__arg1": "2+2"}`,
						},
					},
					{
						ID: "call2",
						FunctionCall: &llms.FunctionCall{
							Name:      "weather",
							Arguments: `{"__arg1": "Seattle"}`,
						},
					},
				},
			},
		},
	}

	actions, finish, err := agent.ParseOutput(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if finish != nil {
		t.Error("expected actions, got finish")
	}
	if len(actions) != 2 {
		t.Errorf("expected 2 actions, got %d", len(actions))
	}
}
