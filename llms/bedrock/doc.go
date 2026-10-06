// Package bedrock provides AWS Bedrock integration for LangChainGo.
//
// # Overview
//
// This package implements LLM client for AWS Bedrock, supporting the model
// providers listed in models_list.go.
//
// # Architecture
//
// The package consists of three layers:
//
//  1. Public API Layer (bedrockllm.go): Exposes bedrock.LLM and bedrock.New() constructor
//  2. Message Processing Layer: Converts llms.MessageContent to provider-specific formats
//  3. Internal Client Layer (internal/bedrockclient): Handles AWS SDK interactions
//
// Two API modes are supported:
//
//   - Legacy API: Model-specific implementations via InvokeModel/InvokeModelWithResponseStream
//   - Converse API: Unified implementation via Converse/ConverseStream (recommended)
//
// # Basic Usage
//
// Create a Bedrock client:
//
//	import "github.com/vxcontrol/langchaingo/llms/bedrock"
//
//	llm, err := bedrock.New(
//	    bedrock.WithModel(bedrock.ModelAnthropicClaudeSonnet45),
//	    bedrock.WithConverseAPI(),
//	)
//
// Generate content:
//
//	messages := []llms.MessageContent{
//	    llms.TextParts(llms.ChatMessageTypeHuman, "Hello!"),
//	}
//	resp, err := llm.GenerateContent(ctx, messages,
//	    llms.WithMaxTokens(1024),
//	)
//
// # Automatic Prompt Caching
//
// For Claude models whose IDs contain claude-opus-4, claude-sonnet-4,
// claude-haiku-4 or a 5.x Opus, Sonnet, Haiku, Fable or Mythos name, automatic
// caching is available:
//
//	llm, err := bedrock.New(
//	    bedrock.WithModel(bedrock.ModelAnthropicClaudeSonnet45),
//	    bedrock.WithConverseAPI(),
//	    bedrock.WithAutomaticCaching(),  // Enable automatic caching
//	)
//
// When enabled, the client marks the last assistant or tool-result message
// before the new user turn with a 5-minute cache point. Converse carries that
// mark on an assistant message only and adds cache points after the system
// prompt and at the end of the final message; the InvokeModel path sends the
// system prompt uncached.
//
// Manual caching (for fine-grained control):
//
//	messages := []llms.MessageContent{
//	    {
//	        Role: llms.ChatMessageTypeAI,
//	        Parts: []llms.ContentPart{
//	            bedrock.WithCacheControl(
//	                llms.TextPart("long context..."),
//	                bedrock.EphemeralCache(),
//	            ),
//	        },
//	    },
//	}
//
// # Tool Calling
//
// Both APIs support tool calling for compatible models:
//
//	tools := []llms.Tool{
//	    {
//	        Type: "function",
//	        Function: &llms.FunctionDefinition{
//	            Name: "get_weather",
//	            Description: "Get weather for location",
//	            Parameters: map[string]any{...},
//	        },
//	    },
//	}
//
//	resp, err := llm.GenerateContent(ctx, messages,
//	    llms.WithTools(tools),
//	)
//
// # Reasoning Support
//
// Claude models with thinking, Nova 2 Lite and GPT OSS take reasoning settings:
//
//	resp, err := llm.GenerateContent(ctx, messages,
//	    llms.WithReasoning(llms.ReasoningMedium, 2048),
//	)
//
//	// Access reasoning content
//	if resp.Choices[0].Reasoning != nil {
//	    fmt.Println(resp.Choices[0].Reasoning.Content)
//	}
//
// # Streaming
//
// Both APIs support streaming responses:
//
//	streamFunc := func(ctx context.Context, chunk streaming.Chunk) error {
//	    switch chunk.Type {
//	    case streaming.ChunkTypeText:
//	        fmt.Print(chunk.Content)
//	    case streaming.ChunkTypeReasoning:
//	        fmt.Println("Thinking:", chunk.Reasoning.Content)
//	    case streaming.ChunkTypeToolCall:
//	        fmt.Println("Tool:", chunk.ToolCall.Name)
//	    }
//	    return nil
//	}
//
//	resp, err := llm.GenerateContent(ctx, messages,
//	    llms.WithStreamingFunc(streamFunc),
//	)
//
// # Supported Models
//
// models_list.go lists the model IDs this package names, with notes on each.
//
// # Error Handling
//
// Errors come back as the AWS SDK returns them. MapError maps them to the
// standardized llms error codes:
//
//	resp, err := llm.GenerateContent(ctx, messages)
//	var llmErr *llms.Error
//	if errors.As(bedrock.MapError(err), &llmErr) {
//	    switch llmErr.Code {
//	    case llms.ErrCodeRateLimit:
//	        // Handle rate limiting
//	    case llms.ErrCodeAuthentication:
//	        // Handle auth errors
//	    }
//	}
//
// See errors.go for complete error mapping.
//
// # AWS Configuration
//
// The client uses AWS SDK v2 configuration:
//
//   - Credentials: From environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) or AWS config
//   - Region: From environment (AWS_REGION) or default config
//   - Custom configuration: Use bedrock.WithClient() with pre-configured bedrockruntime.Client
//
// # Performance Considerations
//
//   - Converse API is recommended for new applications (unified, better error handling)
//   - Streaming reduces latency for interactive applications
//   - A cache checkpoint takes effect only above the model's minimum prefix; the model card names it
//
// # Maintenance
//
// When adding new models:
//  1. Add model constant to models_list.go with documentation
//  2. Update provider detection in internal/bedrockclient/bedrockclient.go if needed
//  3. Add provider-specific implementation in internal/bedrockclient/provider_*.go
//  4. Update tests in bedrockllm_test.go to include new model
//  5. For caching support, add pattern to supportsCaching() method
//
// When updating API:
//  1. Converse API changes go to internal/bedrockclient/bedrockclient_converse.go
//  2. Legacy API changes go to internal/bedrockclient/provider_*.go
//  3. Message processing changes go to bedrockllm.go (processMessages, processMessagesWithCaching)
//  4. Always maintain backward compatibility
//  5. Add integration tests with httprr recording
//
// # Testing
//
// Tests use httprr for HTTP recording/replay:
//
//   - Integration tests: bedrockllm_test.go (replay needs no credentials; recording needs AWS credentials)
//   - Unit tests: bedrockllm_unit_test.go (no credentials needed)
//   - Tool calling: tool_call_test.go and TestAmazonToolCalling* in bedrockllm_test.go
//
// Recording new HTTP interactions:
//
//	HTTPRR_RECORD=. go test -v -run TestName ./llms/bedrock/
//
// Log HTTP traffic:
//
//	HTTPRR_HTTPDEBUG=true go test -v -run TestName ./llms/bedrock/
package bedrock
