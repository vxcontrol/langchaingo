package llms

import (
	"cmp"
	"encoding/json"
	"fmt"

	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

// ToolChoiceKind is what a tool choice asks of the model, apart from the
// spelling the door expects. Doors translate it into their own wire shape.
type ToolChoiceKind int

const (
	ToolChoiceUnset ToolChoiceKind = iota
	ToolChoiceAuto
	ToolChoiceNone
	ToolChoiceAny
	ToolChoiceNamed
)

func (k ToolChoiceKind) String() string {
	switch k {
	case ToolChoiceAuto:
		return "auto"
	case ToolChoiceNone:
		return "none"
	case ToolChoiceAny:
		return "any"
	case ToolChoiceNamed:
		return "a named tool"
	default:
		return "unset"
	}
}

// ClassifyToolChoice reads every spelling the doors accept and reports what was
// asked, naming the tool when the caller picked one.
func ClassifyToolChoice(choice any) (ToolChoiceKind, string) {
	kindOf := func(t string, name string) (ToolChoiceKind, string) {
		switch t {
		case "auto":
			return ToolChoiceAuto, ""
		case "none":
			return ToolChoiceNone, ""
		case "tool", "function":
			if name != "" {
				return ToolChoiceNamed, name
			}
			return ToolChoiceAny, ""
		case "any", "required":
			return ToolChoiceAny, ""
		}
		return ToolChoiceUnset, ""
	}

	switch c := choice.(type) {
	case string:
		return kindOf(c, "")
	case ToolChoice:
		return kindOf(c.Type, functionName(c.Function))
	case *ToolChoice:
		if c == nil {
			return ToolChoiceUnset, ""
		}
		return kindOf(c.Type, functionName(c.Function))
	case map[string]any:
		t, _ := c["type"].(string)
		name, _ := c["name"].(string)
		switch fn := c["function"].(type) {
		case map[string]any:
			name, _ = fn["name"].(string)
		case map[string]string:
			name = fn["name"]
		case FunctionReference:
			name = fn.Name
		case *FunctionReference:
			name = functionName(fn)
		}
		return kindOf(t, name)
	}
	return classifyAsJSON(choice)
}

func classifyAsJSON(choice any) (ToolChoiceKind, string) {
	raw, err := json.Marshal(choice)
	if err != nil {
		return ToolChoiceUnset, ""
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return ToolChoiceUnset, ""
	}
	switch value.(type) {
	case string, map[string]any:
		return ClassifyToolChoice(value)
	}
	return ToolChoiceUnset, ""
}

// DisablesParallelToolUse reports whether a raw map choice asks for at most one
// tool call per turn, Anthropic's disable_parallel_tool_use.
func DisablesParallelToolUse(choice any) bool {
	c, ok := choice.(map[string]any)
	if !ok {
		return false
	}
	disabled, _ := c["disable_parallel_tool_use"].(bool)
	return disabled
}

// ForcesToolUse reports whether a tool choice demands a tool call rather than
// leaving the decision to the model.
func ForcesToolUse(choice any) bool {
	kind, _ := ClassifyToolChoice(choice)
	return kind == ToolChoiceAny || kind == ToolChoiceNamed
}

// ForcedToolName reports whether the choice demands a tool call, and names the
// tool when the caller picked one. An empty name with forced=true means "any
// tool", which every door spells differently.
func ForcedToolName(choice any) (name string, forced bool) {
	kind, name := ClassifyToolChoice(choice)
	switch kind { //nolint:exhaustive // the kinds that leave the choice to the model are not forcing
	case ToolChoiceNamed:
		return name, true
	case ToolChoiceAny:
		return "", true
	}
	return "", false
}

// CheckForcedToolUse refuses a forced tool choice on a Claude model that
// rejects one; a model that only inherits the refusal goes out, recorded in warn.
func CheckForcedToolUse(model string, opts CallOptions, warn *Warnings) error {
	name, forced := ForcedToolName(opts.ToolChoice)
	if !forced || !OffersTools(opts) || !reasoning.ClaudeRejectsForcedToolUse(model) {
		return nil
	}
	asked := cmp.Or(name, spelledChoice(opts.ToolChoice), "any")
	refusal := &reasoning.ErrForcedToolChoiceUnsupported{Model: model, Choice: cmp.Or(name, "any")}
	if warn.KeepRefusal(model, "WithToolChoice", asked, asked, refusal) {
		return refusal
	}
	return nil
}

func spelledChoice(choice any) string {
	switch c := choice.(type) {
	case string:
		return c
	case ToolChoice:
		return c.Type
	case *ToolChoice:
		return c.Type
	case map[string]any:
		spelled, _ := c["type"].(string)
		return spelled
	}
	return ""
}

// OffersTools judges the options as the door sends them: a door that leaves one
// of these sources of tools off the wire passes the options without it.
func OffersTools(opts CallOptions) bool {
	return len(opts.Tools) > 0 || len(opts.Functions) > 0 || ExtraBodyTools(ExtraBody(opts)) > 0
}

func CheckToolCalls(messages []MessageContent) error {
	for _, message := range messages {
		for _, part := range message.Parts {
			if call, ok := part.(ToolCall); ok && call.FunctionCall == nil {
				return NewError(ErrCodeInvalidRequest, "", fmt.Sprintf("tool call %q carries no function to replay", call.ID))
			}
		}
	}
	return nil
}

func functionName(fn *FunctionReference) string {
	if fn == nil {
		return ""
	}
	return fn.Name
}

// HasAssistantPrefill reports whether the conversation ends with an assistant
// turn, which some models reject.
func HasAssistantPrefill(messages []MessageContent) bool {
	if len(messages) == 0 {
		return false
	}
	return messages[len(messages)-1].Role == ChatMessageTypeAI
}

func WithoutSystemMessages(messages []MessageContent) []MessageContent {
	conversation := make([]MessageContent, 0, len(messages))
	for _, message := range messages {
		if message.Role != ChatMessageTypeSystem {
			conversation = append(conversation, message)
		}
	}
	return conversation
}

// CheckClaudeToolChoice refuses the tool choices a Claude model rejects on the
// wire: a forced choice beside manual (budget) thinking, and a forced choice the
// model rejects whatever the thinking. A door that sends only an effort passes
// false for sendsManualThinking.
func CheckClaudeToolChoice(model string, opts CallOptions, sendsManualThinking bool, warn *Warnings) error {
	budget := reasoning.ClaudeClampBudget(model, opts.Reasoning.GetTokens(opts.GetMaxTokens()))
	budgetOnly := reasoning.ClaudeReasoningKindFor(model) == reasoning.ClaudeReasoningBudgetOnly
	budgetThinking := (sendsManualThinking || budgetOnly) &&
		opts.Reasoning.ResolveMode() == ReasoningOn &&
		reasoning.ClaudeSupportsThinking(model) &&
		!reasoning.ResolveClaudeAdaptive(model, opts.Reasoning.Adaptive) &&
		budget > 0
	if budgetThinking && ForcesToolUse(opts.ToolChoice) && OffersTools(opts) {
		return &reasoning.ErrForcedToolUseWithThinking{Model: model}
	}
	return CheckForcedToolUse(model, opts, warn)
}

// CheckClaudePrefill refuses a request whose turns, as the door sends them, end on
// an assistant turn the model rejects as a prefill.
func CheckClaudePrefill(model string, endsOnAssistant bool) error {
	if endsOnAssistant && reasoning.ClaudeRejectsAssistantPrefill(model) {
		return &reasoning.ErrAssistantPrefillUnsupported{Model: model}
	}
	return nil
}
