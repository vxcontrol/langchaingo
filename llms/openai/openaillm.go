package openai

import (
	"context"
	"fmt"

	"github.com/vxcontrol/langchaingo/callbacks"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai/internal/openaiclient"
)

type ChatMessage = openaiclient.ChatMessage

type LLM struct {
	CallbacksHandler callbacks.Handler
	client           *openaiclient.Client
}

const (
	RoleSystem    = "system"
	RoleAssistant = "assistant"
	RoleUser      = "user"
	RoleFunction  = "function"
	RoleTool      = "tool"
)

var _ llms.Model = (*LLM)(nil)

// New returns a new OpenAI LLM.
func New(opts ...Option) (*LLM, error) {
	opt, c, err := newClient(opts...)
	if err != nil {
		return nil, err
	}
	return &LLM{
		client:           c,
		CallbacksHandler: opt.callbackHandler,
	}, err
}

// Call requests a completion for the given prompt.
func (o *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, o, prompt, options...)
}

// Create Text to Speech.
func (o *LLM) GenerateTTS(ctx context.Context, input string, options ...llms.CallOption) ([]byte, error) {
	if input == "" {
		return nil, fmt.Errorf("input is empty")
	}

	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}

	req := &openaiclient.TTSRequest{
		Input:          input,
		Model:          opts.Model,
		Voice:          opts.Voice,
		ResponseFormat: opts.ResponseFormat,
		Speed:          opts.Speed,
	}

	if req.Model != string(openaiclient.TTS1) && req.Model != string(openaiclient.TTS1HD) {
		req.Model = string(openaiclient.TTS1)
	}

	result, err := o.client.CreateTTS(ctx, req)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// GenerateContent implements the Model interface.
func (o *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) { //nolint: lll, cyclop, err113, funlen
	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}

	chatMsgs, err := o.convertMessages(messages)
	if err != nil {
		return nil, err
	}

	req, err := o.createChatRequest(chatMsgs, opts)
	if err != nil {
		return nil, err
	}

	result, err := o.client.CreateChat(ctx, req)
	if err != nil {
		return nil, err
	}
	if len(result.Choices) == 0 {
		return nil, ErrEmptyResponse
	}

	response := o.processResponse(result)

	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, response)
	}
	return response, nil
}

// convertMessages converts LangChain messages to OpenAI chat messages.
func (o *LLM) convertMessages(messages []llms.MessageContent) ([]*ChatMessage, error) {
	chatMsgs := make([]*ChatMessage, 0, len(messages))
	for _, mc := range messages {
		msg := &ChatMessage{MultiContent: mc.Parts}

		if err := o.setMessageRole(msg, mc); err != nil {
			return nil, err
		}

		newParts, toolCalls, toolCallResponses := ExtractToolParts(msg)
		msg.MultiContent = newParts
		msg.ToolCalls = toolCallsFromToolCalls(toolCalls)
		if len(msg.MultiContent) != 0 || len(msg.ToolCalls) != 0 {
			if msg.Role == RoleTool {
				msg.Role = RoleAssistant
			}
			chatMsgs = append(chatMsgs, msg)
		}

		for _, toolCallResponse := range toolCallResponses {
			chatMsgs = append(chatMsgs, &ChatMessage{
				Role:       RoleTool,
				Content:    toolCallResponse.Content,
				Name:       toolCallResponse.Name,
				ToolCallID: toolCallResponse.ToolCallID,
			})
		}
	}

	return chatMsgs, nil
}

// setMessageRole sets the appropriate role for a message and handles special cases.
func (o *LLM) setMessageRole(msg *ChatMessage, mc llms.MessageContent) error {
	switch mc.Role {
	case llms.ChatMessageTypeSystem:
		msg.Role = RoleSystem
	case llms.ChatMessageTypeAI:
		msg.Role = RoleAssistant
	case llms.ChatMessageTypeHuman:
		msg.Role = RoleUser
	case llms.ChatMessageTypeGeneric:
		msg.Role = RoleUser
	case llms.ChatMessageTypeFunction:
		msg.Role = RoleFunction
		return o.handleFunctionMessage(msg, mc)
	case llms.ChatMessageTypeTool:
		msg.Role = RoleTool
		return o.handleToolMessage(mc)
	default:
		return fmt.Errorf("role %v not supported", mc.Role)
	}
	return nil
}

// handleFunctionMessage handles function messages.
func (o *LLM) handleFunctionMessage(msg *ChatMessage, mc llms.MessageContent) error {
	if len(mc.Parts) != 1 {
		return fmt.Errorf("expected exactly one part for role %v, got %v", mc.Role, len(mc.Parts))
	}

	switch p := mc.Parts[0].(type) {
	case llms.ToolCallResponse:
		msg.ToolCallID = p.ToolCallID
		msg.Name = p.Name
		msg.Content = p.Content
	default:
		return fmt.Errorf("expected part of type ToolCallResponse for role %v, got %T",
			mc.Role, mc.Parts[0])
	}

	return nil
}

// handleToolMessage handles tool messages and returns complete tool response messages.
func (o *LLM) handleToolMessage(mc llms.MessageContent) error {
	for _, p := range mc.Parts {
		switch tr := p.(type) {
		case llms.ToolCallResponse:
			if tr.ToolCallID == "" || tr.Name == "" {
				return fmt.Errorf("tool call ID or name is empty for part %v", tr)
			}
		case llms.TextContent:
			// ignore text content, it should be handled on ExtractToolParts call
		default:
			return fmt.Errorf("expected part of type ToolCallResponse for role %v, got %T", mc.Role, tr)
		}
	}

	return nil
}

// createChatRequest creates an OpenAI chat request with the given parameters.
func (o *LLM) createChatRequest(chatMsgs []*ChatMessage, opts llms.CallOptions) (*openaiclient.ChatRequest, error) {
	req := &openaiclient.ChatRequest{
		Model:                  opts.Model,
		StopWords:              opts.StopWords,
		Messages:               chatMsgs,
		StreamingFunc:          opts.StreamingFunc,
		StreamingReasoningFunc: opts.StreamingReasoningFunc,
		Temperature:            opts.Temperature,
		N:                      opts.N,
		FrequencyPenalty:       opts.FrequencyPenalty,
		PresencePenalty:        opts.PresencePenalty,
		MaxCompletionTokens:    opts.MaxTokens,
		ToolChoice:             opts.ToolChoice,
		FunctionCallBehavior:   openaiclient.FunctionCallBehavior(opts.FunctionCallBehavior),
		Seed:                   opts.Seed,
		Metadata:               opts.Metadata,
	}

	if opts.JSONMode {
		req.ResponseFormat = ResponseFormatJSON
	}

	// add tools from functions and tool definitions
	if err := o.addToolsToRequest(req, opts); err != nil {
		return nil, err
	}

	// set response format from client if available
	if o.client.ResponseFormat != nil {
		req.ResponseFormat = o.client.ResponseFormat
	}

	return req, nil
}

// addToolsToRequest adds tools to the request from functions and tool definitions.
func (o *LLM) addToolsToRequest(req *openaiclient.ChatRequest, opts llms.CallOptions) error {
	// add function-based tools (deprecated approach)
	for _, fn := range opts.Functions {
		req.Tools = append(req.Tools, openaiclient.Tool{
			Type: "function",
			Function: openaiclient.FunctionDefinition{
				Name:        fn.Name,
				Description: fn.Description,
				Parameters:  fn.Parameters,
				Strict:      fn.Strict,
			},
		})
	}

	// if opts.Tools is not empty, append them to req.Tools
	for _, tool := range opts.Tools {
		t, err := toolFromTool(tool)
		if err != nil {
			return fmt.Errorf("failed to convert llms tool to openai tool: %w", err)
		}
		req.Tools = append(req.Tools, t)
	}

	return nil
}

// processResponse processes the OpenAI API response into a ContentResponse.
func (o *LLM) processResponse(result *openaiclient.ChatCompletionResponse) *llms.ContentResponse {
	choices := make([]*llms.ContentChoice, len(result.Choices))

	for i, c := range result.Choices {
		choices[i] = &llms.ContentChoice{
			Content:          c.Message.Content,
			ReasoningContent: c.Message.ReasoningContent,
			StopReason:       fmt.Sprint(c.FinishReason),
			GenerationInfo: map[string]any{
				"CompletionTokens": result.Usage.CompletionTokens,
				"PromptTokens":     result.Usage.PromptTokens,
				"TotalTokens":      result.Usage.TotalTokens,
				"ReasoningTokens":  result.Usage.CompletionTokensDetails.ReasoningTokens,
			},
		}

		o.processToolCalls(choices[i], c)
	}

	return &llms.ContentResponse{Choices: choices}
}

// processToolCalls processes tool calls in the response.
func (o *LLM) processToolCalls(choice *llms.ContentChoice, c *openaiclient.ChatCompletionChoice) {
	// legacy function call handling
	if c.FinishReason == "function_call" {
		choice.FuncCall = &llms.FunctionCall{
			Name:      c.Message.FunctionCall.Name,
			Arguments: c.Message.FunctionCall.Arguments,
		}
	}

	for _, tool := range c.Message.ToolCalls {
		choice.ToolCalls = append(choice.ToolCalls, llms.ToolCall{
			ID:   tool.ID,
			Type: string(tool.Type),
			FunctionCall: &llms.FunctionCall{
				Name:      tool.Function.Name,
				Arguments: tool.Function.Arguments,
			},
		})
	}

	// populate legacy single-function call field for backwards compatibility
	if len(choice.ToolCalls) > 0 {
		choice.FuncCall = choice.ToolCalls[0].FunctionCall
	}
}

// CreateEmbedding creates embeddings for the given input texts.
func (o *LLM) CreateEmbedding(ctx context.Context, inputTexts []string) ([][]float32, error) {
	embeddings, err := o.client.CreateEmbedding(ctx, &openaiclient.EmbeddingRequest{
		Input: inputTexts,
		Model: o.client.EmbeddingModel,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create openai embeddings: %w", err)
	}
	if len(embeddings) == 0 {
		return nil, ErrEmptyResponse
	}
	if len(inputTexts) != len(embeddings) {
		return embeddings, ErrUnexpectedResponseLength
	}
	return embeddings, nil
}

// ExtractToolParts extracts the tool parts from a message.
func ExtractToolParts(msg *ChatMessage) ([]llms.ContentPart, []llms.ToolCall, []llms.ToolCallResponse) {
	var content []llms.ContentPart
	var toolCalls []llms.ToolCall
	var toolCallResponses []llms.ToolCallResponse
	for _, part := range msg.MultiContent {
		switch p := part.(type) {
		case llms.ToolCall:
			toolCalls = append(toolCalls, p)
		case llms.ToolCallResponse:
			toolCallResponses = append(toolCallResponses, p)
		case llms.TextContent, llms.ImageURLContent, llms.BinaryContent:
			content = append(content, p)
		default:
			// ignore other parts
		}
	}
	return content, toolCalls, toolCallResponses
}

// toolFromTool converts an llms.Tool to a Tool.
func toolFromTool(t llms.Tool) (openaiclient.Tool, error) {
	tool := openaiclient.Tool{
		Type: openaiclient.ToolType(t.Type),
	}
	switch t.Type {
	case string(openaiclient.ToolTypeFunction):
		tool.Function = openaiclient.FunctionDefinition{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
			Strict:      t.Function.Strict,
		}
	default:
		return openaiclient.Tool{}, fmt.Errorf("tool type %v not supported", t.Type)
	}
	return tool, nil
}

// toolCallsFromToolCalls converts a slice of llms.ToolCall to a slice of ToolCall.
func toolCallsFromToolCalls(tcs []llms.ToolCall) []openaiclient.ToolCall {
	toolCalls := make([]openaiclient.ToolCall, len(tcs))
	for i, tc := range tcs {
		toolCalls[i] = toolCallFromToolCall(tc)
	}
	return toolCalls
}

// toolCallFromToolCall converts an llms.ToolCall to a ToolCall.
func toolCallFromToolCall(tc llms.ToolCall) openaiclient.ToolCall {
	return openaiclient.ToolCall{
		ID:   tc.ID,
		Type: openaiclient.ToolType(tc.Type),
		Function: openaiclient.ToolFunction{
			Name:      tc.FunctionCall.Name,
			Arguments: tc.FunctionCall.Arguments,
		},
	}
}
