package openai

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"github.com/vxcontrol/langchaingo/callbacks"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai/internal/openaiclient"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
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
		Model:          opts.GetModel(),
		Voice:          opts.GetVoice(),
		ResponseFormat: opts.GetResponseFormat(),
		Speed:          opts.GetSpeed(),
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
func (o *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (resp *llms.ContentResponse, err error) { //nolint:lll,nonamedreturns
	// Emit exactly one closing callback for the call: HandleLLMError on any error
	// (transport, config or a structured-output validation failure), otherwise
	// HandleLLMGenerateContentEnd — consistent across every provider adapter.
	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
		defer func() {
			if err != nil {
				o.CallbacksHandler.HandleLLMError(ctx, err)
			} else {
				o.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, resp)
			}
		}()
	}

	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}

	if err := opts.ValidateReasoning(); err != nil {
		return nil, err
	}

	sendsBudget := o.client.UseReasoningMaxTokens && opts.Reasoning.HasExplicitTokens()
	if err := llms.CheckClaudeTurnLimitsOnWire(o.effectiveModel(opts), opts, messages, sendsBudget); err != nil {
		return nil, err
	}

	chatMsgs, err := o.convertMessages(messages, o.effectiveModel(opts))
	if err != nil {
		return nil, err
	}

	warn := &llms.Warnings{}
	reportOpenAIUnread(warn, o.effectiveModel(opts), opts)
	req, err := o.createChatRequest(chatMsgs, opts, warn)
	if err != nil {
		return nil, err
	}

	result, err := o.client.CreateChat(ctx, req)
	if err != nil {
		if result == nil || len(result.Choices) == 0 {
			return nil, err
		}
		return o.partialWithTruncation(result, warn, opts, err)
	}
	if len(result.Choices) == 0 {
		return nil, ErrEmptyResponse
	}

	response := o.processResponse(result, warn)

	if refusal, choice := refusalFrom(result); refusal != nil {
		if opts.StructuredOutput != nil {
			return response, &ErrStructuredOutputRefusal{
				Model:   o.effectiveModel(opts),
				Choice:  choice,
				Refusal: refusal.Message,
				cause:   refusal,
			}
		}
		return response, refusal
	}

	if err := llms.CheckTruncation(response, opts); err != nil {
		return response, err
	}

	// When structured output was requested, validate each normal-final choice
	// against the original schema. The response is still returned alongside the
	// typed error so callers keep usage and diagnostics.
	if err := o.validateStructuredResponse(result, opts); err != nil {
		return response, err
	}

	return response, nil
}

// convertMessages converts LangChain messages to OpenAI chat messages.
func (o *LLM) convertMessages(messages []llms.MessageContent, model string) ([]*ChatMessage, error) {
	chatMsgs := make([]*ChatMessage, 0, len(messages))
	for _, mc := range messages {
		msg := &ChatMessage{MultiContent: mc.Parts}

		if err := o.setMessageRole(msg, mc); err != nil {
			return nil, err
		}

		newParts, toolCalls, toolCallResponses := ExtractToolParts(msg)
		msg.MultiContent = newParts
		msg.ToolCalls = toolCallsFromToolCalls(toolCalls)

		if o.client != nil && o.client.PreserveReasoningContent && msg.Role == RoleAssistant {
			switch {
			case reasoning.ServedByMistral(model):
				if reasoning.ReplaysThinkingInContent(model) {
					msg.Thinking = extractReasoningContent(mc.Parts)
				}
			case len(toolCalls) > 0 || reasoning.ReplaysReasoningOnEveryTurn(model):
				msg.ReasoningContent = extractReasoningContent(mc.Parts)
			}
		}

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
func (o *LLM) createChatRequest(
	chatMsgs []*ChatMessage, opts llms.CallOptions, warn *llms.Warnings,
) (*openaiclient.ChatRequest, error) {
	req := &openaiclient.ChatRequest{
		Model:                opts.GetModel(),
		StopWords:            opts.StopWords,
		Messages:             chatMsgs,
		StreamingFunc:        opts.StreamingFunc,
		Temperature:          opts.Temperature,
		TopK:                 opts.TopK,
		TopP:                 opts.TopP,
		MinP:                 opts.MinP,
		N:                    opts.N,
		FrequencyPenalty:     opts.FrequencyPenalty,
		PresencePenalty:      opts.PresencePenalty,
		RepetitionPenalty:    opts.RepetitionPenalty,
		Verbosity:            opts.Verbosity,
		LogProbs:             opts.LogProbs != nil && *opts.LogProbs,
		TopLogProbs:          derefInt(opts.TopLogProbs),
		ToolChoice:           openaiToolChoice(opts.ToolChoice),
		FunctionCallBehavior: openaiclient.FunctionCallBehavior(opts.FunctionCallBehavior),
		Seed:                 opts.Seed,
		Metadata:             opts.Metadata,
		WebSearchOptions:     webSearchOptionsFromCallOptions(opts.WebSearchOptions),
		ExtraBody:            getExtraBody(&opts),
	}

	model := o.effectiveModel(opts)
	if reasoning.RejectsPenalties(model) {
		const refused = "the door does not send the penalties on this model family"
		addNonZeroChange(warn, "WithFrequencyPenalty", model, refused, req.FrequencyPenalty, nil)
		addNonZeroChange(warn, "WithPresencePenalty", model, refused, req.PresencePenalty, nil)
		req.FrequencyPenalty = nil
		req.PresencePenalty = nil
	}
	if reasoning.RejectsTopK(model) && req.TopK != nil {
		addNonZeroIntChange(warn, "WithTopK", model, refusedByEndpoint, req.TopK, nil)
		req.TopK = nil
	}
	if reasoning.RejectsRepetitionPenalty(model) && req.RepetitionPenalty != nil {
		addNonZeroChange(warn, "WithRepetitionPenalty", model, refusedByEndpoint, req.RepetitionPenalty, nil)
		req.RepetitionPenalty = nil
	}

	if model := o.effectiveModel(opts); reasoning.QwenThinkingRequiresStream(model) {
		if opts.StreamingFunc == nil {
			if opts.Reasoning.ResolveMode() == llms.ReasoningOn {
				return nil, &reasoning.ErrThinkingRequiresStream{Model: model}
			}
			thinkingOff := false
			req.EnableThinking = &thinkingOff
		}
	}

	if model := o.effectiveModel(opts); reasoning.QwenThinkingEnabledByFlag(model) &&
		opts.Reasoning.ResolveMode() == llms.ReasoningOn {
		thinkingOn := true
		req.EnableThinking = &thinkingOn
	}

	if isLegacyMaxTokensField(&opts) || reasoning.UsesLegacyMaxTokens(o.effectiveModel(opts)) {
		req.MaxTokens = opts.MaxTokens
	} else {
		req.MaxCompletionTokens = opts.MaxTokens
	}

	if opts.GetJSONMode() {
		req.SetResponseFormat(ResponseFormatJSON)
	}

	// add tools from functions and tool definitions
	if err := o.addToolsToRequest(req, opts); err != nil {
		return nil, err
	}

	// set response format from client if available
	if o.client.ResponseFormat != nil {
		req.SetResponseFormat(o.client.ResponseFormat)
	}

	// per-call schema-constrained structured output takes precedence over JSONMode
	// and conflicts with a client-level response format.
	if err := o.setStructuredOutput(req, opts); err != nil {
		return nil, err
	}

	wireEffort, err := o.setReasoning(req, opts, warn)
	if err != nil {
		return nil, err
	}
	o.applySamplingPolicy(req, opts, wireEffort, warn)

	return req, nil
}

// effectiveModel resolves the model the request runs on: a per-call model wins,
// then the client default, and finally the package default the client itself
// substitutes on the wire — so capability decisions (reasoning-off, effort clamp,
// temperature pinning) key off the same model the API will actually use.
func (o *LLM) effectiveModel(opts llms.CallOptions) string {
	if m := opts.GetModel(); m != "" {
		return m
	}
	if o.client.Model != "" {
		return o.client.Model
	}
	return openaiclient.DefaultChatModel
}

func wireEffortOf(sent bool, effort llms.ReasoningEffort) string {
	if !sent {
		return ""
	}
	return string(effort)
}

// setReasoning writes the reasoning fields and reports the effort that reached the wire.
func (o *LLM) setReasoning(
	req *openaiclient.ChatRequest, opts llms.CallOptions, warn *llms.Warnings,
) (string, error) {
	model := o.effectiveModel(opts)
	toolsRule := reasoning.EffortToolsFree
	if len(opts.Tools) > 0 {
		toolsRule = reasoning.EffortWithTools(model)
	}

	mode := opts.Reasoning.ResolveMode()
	delegated := opts.Reasoning.DelegatesDepth()
	if delegated {
		mode = llms.ReasoningDefault
	}
	switch mode { //nolint:exhaustive // ReasoningOn is handled by the code after the switch
	case llms.ReasoningDefault:
		if toolsRule == reasoning.EffortToolsDisable {
			o.writeDisableEffort(req)
			reportDelegatedDepth(warn, model, delegated, reasoning.OpenAIDisableEffort)
			return reasoning.OpenAIDisableEffort, nil
		}
		if delegated && reasoning.ThinkingOptIn(model) {
			reportDelegatedDepth(warn, model, delegated, "")
		}
		return "", nil
	case llms.ReasoningOff:
		return "", o.setReasoningOff(req, opts)
	}

	acceptsEffort := reasoning.AcceptsEffortWire(model)
	askedEffort := string(opts.Reasoning.GetEffort(opts.GetMaxTokens()))
	effort := reasoning.OpenAIReasoningCapsFor(model).ClampEffort(askedEffort)
	reasoningEffort := llms.ReasoningEffort(reasoning.ClaudeClampEffort(model, effort, reasoning.ProviderOpenAI))
	reasoningTokens := opts.Reasoning.GetTokens(opts.GetMaxTokens())
	sendsEffort := acceptsEffort && reasoningEffort != llms.ReasoningNone
	if toolsRule != reasoning.EffortToolsFree {
		if !o.sendsBudgetInsteadOfEffort(model, opts, reasoningTokens) {
			return "", &reasoning.ErrEffortWithTools{Model: model, Effort: string(reasoningEffort)}
		}
		sendsEffort = false
	}
	if opts.Reasoning.HasExplicitTokens() && reasoningTokens > 0 &&
		reasoning.DashScopeTakesThinkingBudget(model) {
		req.ThinkingBudget = &reasoningTokens
		reportOpenAIReasoning(warn, model, opts.Reasoning, req)
		return wireEffortOf(true, reasoningEffort), nil
	}
	budget, effortBudget := budgetsFor(model, opts, reasoningEffort, reasoningTokens)
	wire := o.writeEffort(req, sendsEffort, reasoningEffort, budget, effortBudget, warnCtx{model, warn})
	reportOpenAIReasoning(warn, model, opts.Reasoning, req)
	return wire, nil
}

func budgetsFor(
	model string, opts llms.CallOptions, effort llms.ReasoningEffort, tokens int,
) (budget, effortBudget int) {
	if opts.Reasoning.HasExplicitTokens() && tokens > 0 {
		budget = reasoning.ClaudeClampBudget(model, tokens)
	}
	if reasoning.ClaudeSpendsThinkingBudget(model) {
		effortBudget = llms.ReasoningEffortBudget(effort, opts.GetMaxTokens())
	}

	return budget, effortBudget
}

func (o *LLM) sendsBudgetInsteadOfEffort(model string, opts llms.CallOptions, tokens int) bool {
	if !opts.Reasoning.HasExplicitTokens() || tokens <= 0 {
		return false
	}

	return reasoning.DashScopeTakesThinkingBudget(model) || o.client.UseReasoningMaxTokens
}

func (o *LLM) writeEffort(
	req *openaiclient.ChatRequest, sends bool, effort llms.ReasoningEffort, budget, effortBudget int,
	wc warnCtx,
) string {
	if !o.client.ModernReasoningFormat {
		if sends {
			req.ReasoningEffort = &effort
			o.raiseAnswerLimitForBudget(req, effortBudget, wc)
		}
		return wireEffortOf(sends, effort)
	}

	switch {
	case o.client.UseReasoningMaxTokens && budget > 0:
		req.Reasoning = &openaiclient.ReasoningOptions{MaxTokens: budget}
		o.raiseAnswerLimitForBudget(req, budget, wc)
	case sends:
		req.Reasoning = &openaiclient.ReasoningOptions{Effort: effort}
		o.raiseAnswerLimitForBudget(req, effortBudget, wc)
	}
	return wireEffortOf(sends, effort)
}

func (o *LLM) raiseAnswerLimitForBudget(req *openaiclient.ChatRequest, budget int, wc warnCtx) {
	for _, limit := range []**int{&req.MaxCompletionTokens, &req.MaxTokens} {
		if *limit == nil || **limit <= 0 {
			continue
		}
		raised := reasoning.ClaudeMaxTokensForBudget(budget, **limit)
		if raised != **limit {
			wc.sink.Add(llms.Warning{
				Kind: llms.WarningClamp, Option: "WithMaxTokens", Model: wc.model,
				Asked: strconv.Itoa(**limit), Sent: strconv.Itoa(raised),
				Reason: "the answer limit was raised to leave room for the thinking budget",
			})
		}
		*limit = &raised
	}
}

// setReasoningOff sends the model's explicit disable token so a reasoning model
// runs as a plain completion; a model whose thinking cannot be disabled returns
// a typed error.
func (o *LLM) setReasoningOff(req *openaiclient.ChatRequest, opts llms.CallOptions) error {
	model := o.effectiveModel(opts)
	switch reasoning.ResolveOff(model, reasoning.ProviderOpenAI) { //nolint:exhaustive // only OpenAI-relevant wires are handled; others are a no-op
	case reasoning.OffUnsupported:
		return &reasoning.ErrReasoningOffUnsupported{Model: model}
	case reasoning.OffEffortNone:
		o.writeDisableEffort(req)
	case reasoning.OffDisableDashScope:
		thinkingOff := false
		req.EnableThinking = &thinkingOff
	case reasoning.OffDisableThinkingObject:
		req.Thinking = &openaiclient.ThinkingOptions{Type: "disabled"}
	}
	return nil
}

func (o *LLM) writeDisableEffort(req *openaiclient.ChatRequest) {
	none := llms.ReasoningEffort(reasoning.OpenAIDisableEffort)
	if o.client.ModernReasoningFormat {
		req.Reasoning = &openaiclient.ReasoningOptions{Effort: none}
	} else {
		req.ReasoningEffort = &none
	}
}

func (o *LLM) applySamplingPolicy(
	req *openaiclient.ChatRequest, opts llms.CallOptions, wireEffort string, warn *llms.Warnings,
) {
	model := o.effectiveModel(opts)
	before := takeSamplingSnapshot(req)
	reason := samplingReason(model, opts, wireEffort)
	o.enforceSamplingPolicy(req, opts, wireEffort)
	before.report(req, model, reason, warn)
}

func (o *LLM) enforceSamplingPolicy(req *openaiclient.ChatRequest, opts llms.CallOptions, wireEffort string) {
	model := o.effectiveModel(opts)
	if reasoning.RejectsMinP(model) {
		req.MinP = nil
	}
	if reasoning.ClaudeRejectsSampling(model) {
		req.Temperature, req.TopP, req.TopK = nil, nil, nil
		return
	}
	if reasoning.FixesSampling(model) {
		req.Temperature, req.TopP = nil, nil
		req.FrequencyPenalty, req.PresencePenalty = nil, nil
		return
	}

	switch {
	case refusesSamplingWhileThinking(model, opts, wireEffort):
		switch {
		case req.Temperature == nil:
		case reasoning.RejectsSamplingWhileThinking(model):
			req.Temperature = nil
		default:
			temperature := 1.0
			req.Temperature = &temperature
		}
		if req.Temperature != nil || req.TopP == nil ||
			!reasoning.ClaudeKeepsTopPWhileThinking(model, *req.TopP) {
			req.TopP = nil
		}
		req.TopK = nil
		req.FrequencyPenalty = nil
		req.PresencePenalty = nil
		req.LogProbs = false
		req.TopLogProbs = 0
	case ignoresTemperatureWhileThinking(model, opts, wireEffort):
		req.Temperature = nil
	case reasoning.ClaudeMutuallyExclusiveSampling(model) && req.Temperature != nil && req.TopP != nil:
		req.TopP = nil
	}
}

func refusesSamplingWhileThinking(model string, opts llms.CallOptions, wireEffort string) bool {
	if !thinkingRuns(model, opts, wireEffort) {
		return false
	}
	return reasoning.RejectsSamplingWhileThinking(model) || reasoning.ClaudeSupportsThinking(model)
}

func ignoresTemperatureWhileThinking(model string, opts llms.CallOptions, wireEffort string) bool {
	return reasoning.IgnoresTemperatureWhileThinking(model) &&
		thinkingRuns(model, opts, wireEffort) && !extraBodyStopsThinking(opts)
}

func extraBodyStopsThinking(opts llms.CallOptions) bool {
	extra := llms.ExtraBody(opts)
	if thinking, ok := extra["thinking"].(map[string]any); ok && thinking["type"] == "disabled" {
		return true
	}
	return extra["reasoning_effort"] == reasoning.OpenAIDisableEffort
}

// thinkingRuns reports whether the model reasons on this request: an effort
// reached the wire, or none did and the model reasons until told otherwise.
func thinkingRuns(model string, opts llms.CallOptions, wireEffort string) bool {
	if opts.Reasoning.IsDisabled() || !reasoning.IsReasoningModel(model) {
		return false
	}
	if isThinkingOnTheWire(wireEffort) || reasoning.ThinkingMarkedInName(model) {
		return true
	}
	if reasoning.ClaudeSupportsThinking(model) {
		return reasoning.ClaudeThinkingDefaultsOn(model)
	}
	return !reasoning.ThinkingOptIn(model)
}

func isThinkingOnTheWire(wireEffort string) bool {
	return wireEffort != "" && wireEffort != reasoning.OpenAIDisableEffort
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

func refusalFrom(result *openaiclient.ChatCompletionResponse) (*llms.ErrModelRefusal, int) {
	for i, c := range result.Choices {
		if c.Message.Refusal == "" {
			continue
		}
		cached := result.Usage.PromptTokensDetails.CachedTokens
		return &llms.ErrModelRefusal{
			Provider:             "openai",
			Message:              c.Message.Refusal,
			InputTokens:          result.Usage.PromptTokens - cached,
			OutputTokens:         result.Usage.CompletionTokens,
			CacheReadInputTokens: cached,
		}, i
	}
	return nil, 0
}

// processResponse processes the OpenAI API response into a ContentResponse.
func (o *LLM) partialWithTruncation(
	result *openaiclient.ChatCompletionResponse, warn *llms.Warnings, opts llms.CallOptions, cause error,
) (*llms.ContentResponse, error) {
	partial := o.processResponse(result, warn)
	if truncated := llms.CheckTruncation(partial, opts); truncated != nil {
		return partial, errors.Join(cause, truncated)
	}

	return partial, cause
}

func (o *LLM) processResponse(
	result *openaiclient.ChatCompletionResponse, warn *llms.Warnings,
) *llms.ContentResponse {
	choices := make([]*llms.ContentChoice, len(result.Choices))

	for i, c := range result.Choices {
		stopReason := string(c.FinishReason)
		choices[i] = &llms.ContentChoice{
			Content:        c.Message.Content,
			Reasoning:      o.processReasoning(c.Message.ReasoningContent),
			StopReason:     stopReason,
			Truncated:      llms.IsTruncated(stopReason),
			GenerationInfo: o.processUsage(&result.Usage),
		}

		// Surface a Structured Outputs refusal so callers can tell it apart from a
		// schema-valid answer without treating it as a validation failure.
		if c.Message.Refusal != "" {
			choices[i].GenerationInfo["Refusal"] = c.Message.Refusal
		}

		o.processToolCalls(choices[i], c)
	}

	return &llms.ContentResponse{Choices: choices, Warnings: warn.List()}
}

func (o *LLM) processUsage(usage *openaiclient.ChatUsage) map[string]any {
	info := map[string]any{
		"CompletionTokens":  usage.CompletionTokens,
		"PromptTokens":      usage.PromptTokens,
		"TotalTokens":       usage.TotalTokens,
		"ReasoningTokens":   usage.CompletionTokensDetails.ReasoningTokens,
		"PromptAudioTokens": usage.PromptTokensDetails.AudioTokens,
		// Standardized fields for cross-provider compatibility
		"PromptCachedTokens":                 usage.PromptTokensDetails.CachedTokens,
		"CacheReadInputTokens":               usage.PromptTokensDetails.CachedTokens,
		"CacheCreationInputTokens":           usage.PromptTokensDetails.CacheWriteTokens,
		"CompletionAudioTokens":              usage.CompletionTokensDetails.AudioTokens,
		"CompletionReasoningTokens":          usage.CompletionTokensDetails.ReasoningTokens,
		"CompletionAcceptedPredictionTokens": usage.CompletionTokensDetails.AcceptedPredictionTokens,
		"CompletionRejectedPredictionTokens": usage.CompletionTokensDetails.RejectedPredictionTokens,
	}
	// Special fields for OpenRouter provider
	for key, cost := range map[string]*float64{
		"UpstreamInferencePromptCost":      usage.CostDetails.UpstreamInferencePromptCost,
		"UpstreamInferenceCompletionsCost": usage.CostDetails.UpstreamInferenceCompletionsCost,
	} {
		if cost != nil {
			info[key] = *cost
		}
	}

	return info
}

// processReasoning processes reasoning content in the response.
func (o *LLM) processReasoning(reasoningContent string) *reasoning.ContentReasoning {
	if reasoningContent == "" {
		return nil
	}

	return &reasoning.ContentReasoning{
		Content:   reasoningContent,
		Signature: nil, // not supported yet for OpenAI compatible providers
	}
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
		return nil, ErrUnexpectedResponseLength
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
		case llms.TextContent:
			if p.Text == "" {
				continue
			}
			content = append(content, p)
		case llms.ImageURLContent, llms.BinaryContent:
			content = append(content, p)
		default:
			// ignore other parts
		}
	}
	return content, toolCalls, toolCallResponses
}

// extractReasoningContent extracts reasoning content from message parts.
// It returns the first non-empty reasoning content found in TextContent parts.
func extractReasoningContent(parts []llms.ContentPart) string {
	for _, part := range parts {
		if tc, ok := part.(llms.TextContent); ok {
			if tc.Reasoning != nil && tc.Reasoning.Content != "" {
				return tc.Reasoning.Content
			}
		}
	}
	return ""
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
	for idx, tc := range tcs {
		toolCalls[idx] = toolCallFromToolCall(tc)
	}
	return toolCalls
}

// toolCallFromToolCall converts an llms.ToolCall to a ToolCall.
func toolCallFromToolCall(tc llms.ToolCall) openaiclient.ToolCall {
	toolType := openaiclient.ToolType(tc.Type)
	if toolType == "" {
		toolType = openaiclient.ToolTypeFunction
	}
	return openaiclient.ToolCall{
		ID:   tc.ID,
		Type: toolType,
		Function: openaiclient.ToolFunction{
			Name:      tc.FunctionCall.Name,
			Arguments: tc.FunctionCall.Arguments,
		},
	}
}

// webSearchOptionsFromCallOptions converts llms.WebSearchOptions to openaiclient.WebSearchOptions.
func webSearchOptionsFromCallOptions(opts *llms.WebSearchOptions) *openaiclient.WebSearchOptions {
	if opts == nil {
		return nil
	}
	result := &openaiclient.WebSearchOptions{
		SearchContextSize: opts.SearchContextSize,
	}
	if opts.UserLocation != nil {
		result.UserLocation = &openaiclient.UserLocation{
			Type: opts.UserLocation.Type,
		}
		if opts.UserLocation.Approximate != nil {
			result.UserLocation.Approximate = &openaiclient.ApproximateLocation{
				Country: opts.UserLocation.Approximate.Country,
				City:    opts.UserLocation.Approximate.City,
				Region:  opts.UserLocation.Approximate.Region,
			}
		}
	}
	return result
}

func openaiToolChoice(choice any) any {
	switch kind, name := llms.ClassifyToolChoice(choice); kind {
	case llms.ToolChoiceNamed:
		return llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: name}}
	case llms.ToolChoiceAny:
		return "required"
	case llms.ToolChoiceAuto:
		return "auto"
	case llms.ToolChoiceNone:
		return "none"
	default:
		return choice
	}
}

func derefInt(i *int) int {
	if i == nil {
		return 0
	}
	return *i
}
