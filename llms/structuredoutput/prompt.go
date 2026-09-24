package structuredoutput

import (
	"encoding/json"
	"strings"
)

// A door whose vendor cannot constrain its output to a JSON Schema carries the
// schema in the prompt instead. These texts are what the model reads; recorded
// requests hold them byte for byte, so a change means re-recording those tests.
const (
	// SchemaInstruction asks for exactly one JSON value matching the schema that
	// follows it, and overrides any earlier instruction about format or style.
	SchemaInstruction = "Reply with exactly one JSON value that validates against the JSON Schema below. " +
		"This overrides any earlier instruction about format or style: no Markdown, no code fences, " +
		"no headings, and no text before or after the JSON. The whole reply is parsed by a JSON parser."
	// ToolsNote keeps tool calls open when tools are offered alongside the schema.
	ToolsNote = "You may still call the tools you are offered; this applies to your final answer."
)

// PromptInstruction returns the instruction that carries schema in the prompt,
// with ToolsNote when the request also offers tools.
func PromptInstruction(schema json.RawMessage, withTools bool) string {
	instruction := SchemaInstruction
	if withTools {
		instruction += " " + ToolsNote
	}
	return instruction + "\nJSON Schema:\n" + string(schema)
}

// UnwrapFencedJSON returns the body of a text that is exactly one Markdown code
// fence, untagged or tagged json, which a model often adds despite the prompt
// instruction. Anything else, such as prose around the fence or a second fence,
// is returned unchanged for validation to judge.
func UnwrapFencedJSON(text string) string {
	trimmed := strings.TrimSpace(text)
	if len(trimmed) < 6 || !strings.HasPrefix(trimmed, "```") || !strings.HasSuffix(trimmed, "```") {
		return text
	}
	header, body, ok := strings.Cut(trimmed[3:len(trimmed)-3], "\n")
	if !ok || containsFenceLine(body) {
		return text
	}
	if tag := strings.TrimSpace(header); tag != "" && !strings.EqualFold(tag, "json") {
		return text
	}
	return strings.TrimSpace(body)
}

// containsFenceLine reports a line that opens or closes a code fence. A JSON
// string cannot hold a raw newline, so backticks inside one never start a line.
func containsFenceLine(body string) bool {
	for line := range strings.Lines(body) {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			return true
		}
	}
	return false
}
