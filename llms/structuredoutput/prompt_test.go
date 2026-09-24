package structuredoutput

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPromptInstruction(t *testing.T) {
	t.Parallel()

	schema := json.RawMessage(`{"type":"object"}`)
	assert.Equal(t, SchemaInstruction+"\nJSON Schema:\n"+`{"type":"object"}`, PromptInstruction(schema, false))
	assert.Equal(t, SchemaInstruction+" "+ToolsNote+"\nJSON Schema:\n"+`{"type":"object"}`, PromptInstruction(schema, true))
	// DeepSeek refuses json_object unless a message mentions JSON; the instruction does.
	assert.Contains(t, strings.ToLower(PromptInstruction(schema, false)), "json")
}

func TestUnwrapFencedJSON(t *testing.T) {
	t.Parallel()

	for text, want := range map[string]string{
		"```json\n{\"a\":1}\n```":            `{"a":1}`,
		"```JSON\n{\"a\":1}\n```":            `{"a":1}`,
		"  ```\n[1, 2]\n```  ":               "[1, 2]",
		"```json\n{\n  \"a\": 1\n}\n```":     "{\n  \"a\": 1\n}",
		`{"a":1}`:                            `{"a":1}`,
		"```json {\"a\":1}```":               "```json {\"a\":1}```",
		"```go\nfmt.Println()\n```":          "```go\nfmt.Println()\n```",
		"note\n```json\n{}\n```":             "note\n```json\n{}\n```",
		"```json\n{}\n```\n```json\n{}\n```": "```json\n{}\n```\n```json\n{}\n```",
		"``````":                             "``````",
		"```json\r\n{\"a\":1}\r\n```":        `{"a":1}`,
		"``` json\n{\"a\":1}\n```":           `{"a":1}`,
		"```json\n{\"a\":\"```\"}\n```":      "{\"a\":\"```\"}",
		"```json\n```":                       "",
		"```json\n\n```":                     "",
		"````json\n{\"a\":1}\n````":          "````json\n{\"a\":1}\n````",
	} {
		assert.Equal(t, want, UnwrapFencedJSON(text), "%q", text)
	}
}
