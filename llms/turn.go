package llms

import (
	"cmp"

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

// CheckClaudeTurnLimits refuses the two turns a Claude model rejects on the
// wire: manual (budget) thinking combined with a forced tool choice, and a
// conversation that ends on an assistant turn.
func CheckClaudeTurnLimits(model string, opts CallOptions, messages []MessageContent, warn *Warnings) error {
	return CheckClaudeTurnLimitsOnWire(model, opts, messages, true, warn)
}

// CheckClaudeTurnLimitsOnWire is CheckClaudeTurnLimits for a door that knows
// whether its own request carries a manual thinking budget; a door that sends
// only an effort passes false.
func CheckClaudeTurnLimitsOnWire(
	model string,
	opts CallOptions,
	messages []MessageContent,
	sendsManualThinking bool,
	warn *Warnings,
) error {
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
	if err := CheckForcedToolUse(model, opts, warn); err != nil {
		return err
	}

	if reasoning.ClaudeRejectsAssistantPrefill(model) && HasAssistantPrefill(messages) {
		return &reasoning.ErrAssistantPrefillUnsupported{Model: model}
	}
	return nil
}
