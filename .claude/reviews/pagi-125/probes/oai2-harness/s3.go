package openai_test

import (
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

func scenarios() []scen {
	rfA := &openai.ResponseFormat{Type: "json_schema", JSONSchema: &openai.ResponseFormatJSONSchema{Name: "a", Strict: true, Schema: &openai.ResponseFormatJSONSchemaProperty{
		Type: "object", Properties: map[string]*openai.ResponseFormatJSONSchemaProperty{"city": {Type: "string"}}, Required: []string{"city"}}}}
	extraB := openai.WithExtraBody(map[string]any{"response_format": map[string]any{"type": "json_schema", "json_schema": map[string]any{
		"name": "b", "strict": true, "schema": map[string]any{"type": "object", "properties": map[string]any{"score": map[string]any{"type": "number"}}, "required": []any{"score"}, "additionalProperties": false}}}})
	return []scen{
		{name: "client-schema-A+extra-schema-B", model: "gpt-4o", cliOpts: []openai.Option{openai.WithResponseFormat(rfA)}, opts: []llms.CallOption{extraB}},
	}
}
