package llms

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestForcesToolUseAcceptsEveryFormOfTheChoice(t *testing.T) {
	t.Parallel()

	named := ToolChoice{Type: "tool", Function: &FunctionReference{Name: "calc"}}
	for _, choice := range []any{
		"any", "tool", "required",
		ToolChoice{Type: "any"}, named,
		&ToolChoice{Type: "any"}, &named,
		map[string]any{"type": "any"},
		map[string]any{"type": "tool", "name": "calc"},
		ToolChoice{Type: "function", Function: &FunctionReference{Name: "calc"}},
		&ToolChoice{Type: "function", Function: &FunctionReference{Name: "calc"}},
		map[string]any{"type": "function", "function": map[string]any{"name": "calc"}},
	} {
		if !ForcesToolUse(choice) {
			t.Errorf("%#v demands a tool call", choice)
		}
	}

	var absent *ToolChoice
	for _, choice := range []any{
		nil, absent, "auto", "none", "",
		ToolChoice{Type: "auto"}, &ToolChoice{Type: "none"},
		map[string]any{"type": "auto"}, map[string]any{},
		42,
	} {
		if ForcesToolUse(choice) {
			t.Errorf("%#v leaves the decision to the model", choice)
		}
	}
}

func TestForcedToolNameReadsEveryNotation(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name   string
		choice any
		want   string
		forced bool
	}{
		{"anthropic named", ToolChoice{Type: "tool", Function: &FunctionReference{Name: "calc"}}, "calc", true},
		{"anthropic named map", map[string]any{"type": "tool", "name": "calc"}, "calc", true},
		{"anthropic any", ToolChoice{Type: "any"}, "", true},
		{"openai named", ToolChoice{Type: "function", Function: &FunctionReference{Name: "calc"}}, "calc", true},
		{"openai named pointer", &ToolChoice{Type: "function", Function: &FunctionReference{Name: "calc"}}, "calc", true},
		{"openai named map", map[string]any{"type": "function", "function": map[string]any{"name": "calc"}}, "calc", true},
		{"openai required", "required", "", true},
		{"auto", ToolChoice{Type: "auto"}, "", false},
		{"none", "none", "", false},
		{"absent", nil, "", false},
	} {
		name, forced := ForcedToolName(tc.choice)
		if name != tc.want || forced != tc.forced {
			t.Errorf("%s: ForcedToolName() = (%q, %v), want (%q, %v)",
				tc.name, name, forced, tc.want, tc.forced)
		}
	}
}

func TestANamedChoiceKeepsItsToolWhateverTheFunctionShape(t *testing.T) {
	t.Parallel()

	for name, function := range map[string]any{
		"map[string]string":  map[string]string{"name": "b"},
		"FunctionReference":  FunctionReference{Name: "b"},
		"*FunctionReference": &FunctionReference{Name: "b"},
		"map[string]any":     map[string]any{"name": "b"},
	} {
		kind, tool := ClassifyToolChoice(map[string]any{"type": "function", "function": function})
		if kind != ToolChoiceNamed || tool != "b" {
			t.Errorf("%s: got (%v, %q), want the named tool b", name, kind, tool)
		}
	}
}

func TestAForcedChoiceIsRefusedOnAClaudeModelThatRejectsIt(t *testing.T) {
	t.Parallel()

	tools := []Tool{{Type: "function", Function: &FunctionDefinition{Name: "echo"}}}
	for _, choice := range []any{"required", map[string]any{"type": "tool", "name": "echo"}} {
		err := CheckClaudeToolChoice("us.anthropic.claude-fable-5-1",
			CallOptions{Tools: tools, ToolChoice: choice}, true, nil)
		var refused *reasoning.ErrForcedToolChoiceUnsupported
		if !errors.As(err, &refused) {
			t.Errorf("%v: got %v, want ErrForcedToolChoiceUnsupported", choice, err)
		}
	}
	if err := CheckClaudeToolChoice("us.anthropic.claude-fable-5-1",
		CallOptions{Tools: tools, ToolChoice: "auto"}, true, nil); err != nil {
		t.Errorf("auto must pass, got %v", err)
	}
	if err := CheckClaudeToolChoice("us.anthropic.claude-fable-5-1",
		CallOptions{ToolChoice: "required"}, true, nil); err != nil {
		t.Errorf("with no tools the choice never reaches the wire, got %v", err)
	}
	thinking := &ReasoningConfig{Mode: ReasoningOn, Tokens: 2048}
	if err := CheckClaudeToolChoice("claude-sonnet-4-5",
		CallOptions{Reasoning: thinking, ToolChoice: "required"}, true, nil); err != nil {
		t.Errorf("budget thinking with no tools sends no choice to refuse, got %v", err)
	}
	var withThinking *reasoning.ErrForcedToolUseWithThinking
	if err := CheckClaudeToolChoice("claude-sonnet-4-5",
		CallOptions{Reasoning: thinking, Tools: tools, ToolChoice: "required"}, true, nil); !errors.As(err, &withThinking) {
		t.Errorf("budget thinking with a tool to force, got %v, want ErrForcedToolUseWithThinking", err)
	}
	functions := []FunctionDefinition{{Name: "echo"}}
	err := CheckClaudeToolChoice("claude-opus-5-5",
		CallOptions{Functions: functions, ToolChoice: "required"}, true, nil)
	var refused *reasoning.ErrForcedToolChoiceUnsupported
	if !errors.As(err, &refused) {
		t.Errorf("the openai door sends legacy functions as tools, got %v", err)
	}
}

type namedChoice string

func TestAChoiceIsReadInTheJSONItGoesOutAs(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		choice any
		kind   ToolChoiceKind
		tool   string
	}{
		"raw JSON string":    {json.RawMessage(`"required"`), ToolChoiceAny, ""},
		"raw JSON object":    {json.RawMessage(`{"type":"function","function":{"name":"b"}}`), ToolChoiceNamed, "b"},
		"string map":         {map[string]string{"type": "none"}, ToolChoiceNone, ""},
		"named string type":  {namedChoice("auto"), ToolChoiceAuto, ""},
		"a value of no form": {42, ToolChoiceUnset, ""},
	} {
		kind, tool := ClassifyToolChoice(tc.choice)
		if kind != tc.kind || tool != tc.tool {
			t.Errorf("%s: got (%v, %q), want (%v, %q)", name, kind, tool, tc.kind, tc.tool)
		}
	}
}

func TestToolsInTheExtraBodyAreCountedAsTheWireCarriesThem(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		tools any
		count int
	}{
		"raw JSON":      {json.RawMessage(`[{"type":"function","function":{"name":"a"}}]`), 1},
		"raw JSON none": {json.RawMessage(`[]`), 0},
		"typed tools":   {[]Tool{{Type: "function"}, {Type: "function"}}, 2},
		"decoded JSON":  {[]any{map[string]any{"type": "function"}}, 1},
	} {
		if got := ExtraBodyTools(map[string]any{"tools": tc.tools}); got != tc.count {
			t.Errorf("%s: got %d tools, want %d", name, got, tc.count)
		}
	}
}
