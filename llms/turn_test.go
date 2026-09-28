package llms

import (
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
		err := CheckClaudeTurnLimits("us.anthropic.claude-fable-5-1",
			CallOptions{Tools: tools, ToolChoice: choice}, nil)
		var refused *reasoning.ErrForcedToolChoiceUnsupported
		if !errors.As(err, &refused) {
			t.Errorf("%v: got %v, want ErrForcedToolChoiceUnsupported", choice, err)
		}
	}
	if err := CheckClaudeTurnLimits("us.anthropic.claude-fable-5-1", CallOptions{Tools: tools, ToolChoice: "auto"}, nil); err != nil {
		t.Errorf("auto must pass, got %v", err)
	}
	if err := CheckClaudeTurnLimits("us.anthropic.claude-fable-5-1", CallOptions{ToolChoice: "required"}, nil); err != nil {
		t.Errorf("with no tools the choice never reaches the wire, got %v", err)
	}
}
