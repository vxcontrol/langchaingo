package openai

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai/internal/openaiclient"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func (o *LLM) takesResponses(model string, opts llms.CallOptions) bool {
	if !o.servedByOpenAI() {
		return false
	}
	mode := reasoning.ThinkingDefault
	switch o.toolTurnMode(model, opts) { //nolint:exhaustive // the default mode keeps ThinkingDefault
	case llms.ReasoningOff:
		mode = reasoning.ThinkingOff
	case llms.ReasoningOn:
		mode = reasoning.ThinkingBudget
	default:
		if len(opts.StopWords) > 0 {
			mode = reasoning.ThinkingOff
		}
	}
	return reasoning.OpenAITakesResponses(model, len(opts.Tools)+len(opts.Functions) > 0, mode)
}

func (o *LLM) toolTurnMode(model string, opts llms.CallOptions) llms.ReasoningMode {
	switch on, off := llms.ExtraBodyThinking(model, "api.openai.com", llms.ExtraBody(opts)); {
	case off:
		return llms.ReasoningOff
	case on:
		return llms.ReasoningOn
	case opts.Reasoning.DelegatesDepth() && !o.claudeThinksOnlyAtAnAskedDepth(model):
		return llms.ReasoningDefault
	}
	return opts.Reasoning.ResolveMode()
}

func (o *LLM) send(
	ctx context.Context, messages []llms.MessageContent, req *openaiclient.ChatRequest, warn *llms.Warnings, responses bool,
) (*openaiclient.ChatCompletionResponse, error) {
	if !responses {
		return o.client.CreateChat(ctx, req)
	}
	input, err := responsesInput(messages)
	if err != nil {
		return nil, err
	}
	if req.Model == "" {
		req.Model = o.effectiveModel(llms.CallOptions{})
	}
	resp, err := o.client.CreateResponse(ctx, responsesRequest(req, input, warn))
	if resp == nil {
		return nil, err
	}
	return resp.ChatResponse(), err
}

func responsesRequest(req *openaiclient.ChatRequest, input []any, warn *llms.Warnings) *openaiclient.ResponsesRequest {
	r := &openaiclient.ResponsesRequest{
		Model: req.Model, Input: input, MaxOutputTokens: req.MaxCompletionTokens,
		Temperature: req.Temperature, TopP: req.TopP, Metadata: req.Metadata, StreamingFunc: req.StreamingFunc,
		PromptCacheKey: req.PromptCacheKey, PromptCacheOptions: req.PromptCacheOptions,
	}
	if r.MaxOutputTokens == nil {
		r.MaxOutputTokens = req.MaxTokens
	}
	for _, tool := range req.Tools {
		r.Tools = append(r.Tools, openaiclient.ResponsesTool{
			Type: "function", Name: tool.Function.Name, Description: tool.Function.Description,
			Parameters: tool.Function.Parameters, Strict: tool.Function.Strict,
		})
	}
	r.ToolChoice = responsesToolChoice(req.ToolChoice)
	switch {
	case req.ReasoningEffort != nil:
		r.Reasoning = &openaiclient.ResponsesReasoning{Effort: string(*req.ReasoningEffort)}
	case req.Reasoning != nil && req.Reasoning.Effort != "":
		r.Reasoning = &openaiclient.ResponsesReasoning{Effort: string(req.Reasoning.Effort)}
	}
	if format := req.ResponsesFormat(); format != nil || req.Verbosity != nil {
		r.Text = &openaiclient.ResponsesText{Format: format}
		if req.Verbosity != nil {
			r.Text.Verbosity = *req.Verbosity
		}
	}
	reportChatOnlyFields(req, warn)
	r.ExtraBody = responsesExtraBody(req, r, warn)
	return r
}

var responsesFields = []string{
	"access_programs", "background", "context_management", "conversation", "include", "input", "instructions",
	"max_output_tokens", "max_tool_calls", "metadata", "model", "moderation", "parallel_tool_calls",
	"previous_response_id", "prompt", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "reasoning",
	"safety_identifier", "service_tier", "store", "stream", "stream_options", "temperature", "text", "tool_choice",
	"tools", "top_logprobs", "top_p", "truncation", "user",
}

func responsesExtraBody(req *openaiclient.ChatRequest, r *openaiclient.ResponsesRequest, warn *llms.Warnings) map[string]any {
	kept := map[string]any{}
	for _, key := range slices.Sorted(maps.Keys(req.ExtraBody)) {
		effort, isEffort := req.ExtraBody[key].(string)
		switch {
		case key == "reasoning_effort" && isEffort:
			r.Reasoning = &openaiclient.ResponsesReasoning{Effort: effort}
		case slices.Contains(responsesFields, key):
			kept[key] = req.ExtraBody[key]
		default:
			warn.Add(llms.Warning{
				Kind: llms.WarningDrop, Option: "WithExtraBody", Model: req.Model, Asked: key,
				Reason: "the Responses API has no such field",
			})
		}
	}
	return kept
}

func responsesToolChoice(choice any) any {
	switch kind, name := llms.ClassifyToolChoice(choice); kind {
	case llms.ToolChoiceNamed:
		return map[string]any{"type": "function", "name": name}
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

func reportChatOnlyFields(req *openaiclient.ChatRequest, warn *llms.Warnings) {
	const reason = "the Responses API has no such field"
	choices := req.N
	if choices != nil && *choices == 1 {
		choices = nil
	}
	for _, field := range []struct{ option, asked string }{
		{"WithN", askedInt(choices)},
		{"WithSeed", askedInt(req.Seed)},
		{"WithFrequencyPenalty", askedFloat(req.FrequencyPenalty)},
		{"WithPresencePenalty", askedFloat(req.PresencePenalty)},
		{"WithRepetitionPenalty", askedFloat(req.RepetitionPenalty)},
		{"WithTopK", askedInt(req.TopK)},
		{"WithMinP", askedFloat(req.MinP)},
	} {
		if field.asked != "" {
			warn.Add(llms.Warning{Kind: llms.WarningDrop, Option: field.option, Model: req.Model, Asked: field.asked, Reason: reason})
		}
	}
	if req.LogProbs || req.TopLogProbs > 0 {
		warn.Add(llms.Warning{Kind: llms.WarningDrop, Option: "WithLogProbs", Model: req.Model, Asked: "true", Reason: reason})
	}
	if req.WebSearchOptions != nil {
		warn.Add(llms.Warning{Kind: llms.WarningDrop, Option: "WithWebSearchOptions", Model: req.Model, Asked: "set", Reason: reason})
	}
}

func askedInt(v *int) string {
	if v == nil {
		return ""
	}
	return strconv.Itoa(*v)
}

func askedFloat(v *float64) string {
	if v == nil {
		return ""
	}
	return strconv.FormatFloat(*v, 'g', -1, 64)
}

func responsesInput(messages []llms.MessageContent) ([]any, error) {
	var input []any
	for _, msg := range messages {
		switch msg.Role {
		case llms.ChatMessageTypeSystem, llms.ChatMessageTypeHuman, llms.ChatMessageTypeGeneric:
			content, err := responsesContent(msg.Parts)
			if err != nil {
				return nil, err
			}
			role := RoleUser
			if msg.Role == llms.ChatMessageTypeSystem {
				role = RoleSystem
			}
			input = append(input, openaiclient.ResponsesMessage{Type: "message", Role: role, Content: content})
		case llms.ChatMessageTypeAI:
			input = append(input, assistantItems(msg)...)
		case llms.ChatMessageTypeTool:
			for _, part := range msg.Parts {
				if result, ok := part.(llms.ToolCallResponse); ok {
					input = append(input, openaiclient.ResponsesFunctionCallOutput{
						Type: "function_call_output", CallID: result.ToolCallID, Output: result.Content,
					})
				}
			}
		default:
			return nil, fmt.Errorf("role %v not supported by the Responses API", msg.Role)
		}
	}
	return input, nil
}

func responsesContent(parts []llms.ContentPart) (any, error) {
	if len(parts) == 1 {
		if text, ok := parts[0].(llms.TextContent); ok {
			return text.Text, nil
		}
	}
	content := make([]openaiclient.ResponsesInputPart, 0, len(parts))
	for _, part := range parts {
		switch p := part.(type) {
		case llms.TextContent:
			content = append(content, openaiclient.ResponsesInputPart{Type: "input_text", Text: p.Text})
		case llms.ImageURLContent:
			content = append(content, openaiclient.ResponsesInputPart{Type: "input_image", ImageURL: p.URL, Detail: p.Detail})
		case llms.BinaryContent:
			if !strings.HasPrefix(strings.ToLower(p.MIMEType), "image/") {
				return nil, fmt.Errorf("%w: binary content of type %q", ErrUnsupportedContentType, p.MIMEType)
			}
			content = append(content, openaiclient.ResponsesInputPart{Type: "input_image", ImageURL: p.String()})
		default:
			return nil, fmt.Errorf("%w: %T", ErrUnsupportedContentType, part)
		}
	}
	return content, nil
}

func assistantItems(msg llms.MessageContent) []any {
	calls, texts := 0, 0
	for _, part := range msg.Parts {
		switch p := part.(type) {
		case llms.ToolCall:
			calls++
		case llms.TextContent:
			if p.Text != "" {
				texts++
			}
		}
	}

	var out []any
	var later []reasoning.Block
	sent := 0
	for _, part := range msg.Parts {
		switch p := part.(type) {
		case llms.TextContent:
			for _, block := range p.Reasoning.Sequence() {
				switch {
				case block.ID == "":
				case block.AfterToolCalls > sent:
					later = append(later, block)
				default:
					out = append(out, reasoningItem(block))
				}
			}
			if p.Text != "" {
				texts--
				out = append(out, openaiclient.ResponsesMessage{
					Type: "message", Role: RoleAssistant, Content: p.Text, Phase: phaseOf(p.Phase, calls > 0 || texts > 0),
				})
			}
		case llms.ToolCall:
			out = append(out, openaiclient.ResponsesFunctionCall{
				Type: "function_call", CallID: p.ID, Name: p.FunctionCall.Name, Arguments: p.FunctionCall.Arguments,
			})
			sent++
			for len(later) > 0 && later[0].AfterToolCalls <= sent {
				out, later = append(out, reasoningItem(later[0])), later[1:]
			}
		}
	}
	for _, block := range later {
		out = append(out, reasoningItem(block))
	}
	return out
}

func phaseOf(phase string, moreFollows bool) string {
	switch {
	case phase != "":
		return phase
	case moreFollows:
		return "commentary"
	}
	return "final_answer"
}

func reasoningItem(block reasoning.Block) openaiclient.ResponsesReasoningItem {
	summary := []openaiclient.ResponsesSummary{}
	if block.Text != "" {
		summary = append(summary, openaiclient.ResponsesSummary{Type: "summary_text", Text: block.Text})
	}
	return openaiclient.ResponsesReasoningItem{
		Type: "reasoning", ID: block.ID, Summary: summary, EncryptedContent: string(block.Redacted),
	}
}
