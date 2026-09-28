package structuredoutput_test

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/structuredoutput"
)

func TestProbeSchemaLoader(t *testing.T) {
	for _, s := range []string{
		`{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","additionalProperties":false,"properties":{"a":{"type":"string"}}}`,
		`{"$schema":"http://json-schema.org/draft-07/schema","type":"object","additionalProperties":false,"properties":{"a":{"type":"string"}}}`,
		`{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false,"properties":{"a":{"type":"string"}}}`,
		`{"$schema":"https://json-schema.org/draft/2019-09/schema","type":"object","additionalProperties":false,"properties":{"a":{"type":"string"}}}`,
		`{"$schema":"http://json-schema.org/draft-04/schema#","type":"object","additionalProperties":false,"properties":{"a":{"type":"string"}}}`,
		`{"$id":"https://example.com/person.schema.json","type":"object","additionalProperties":false,"properties":{"a":{"$ref":"#/$defs/x"}},"$defs":{"x":{"type":"string"}}}`,
		`{"$id":"https://example.com/person.schema.json","type":"object","additionalProperties":false,"properties":{"a":{"$ref":"https://example.com/person.schema.json#/$defs/x"}},"$defs":{"x":{"type":"string"}}}`,
	} {
		_, err := structuredoutput.Compile(json.RawMessage(s))
		o := llms.CallOptions{JSONMode: true, StructuredOutput: &llms.StructuredOutputConfig{Name: "s", Schema: json.RawMessage(s)}}
		err2 := o.ValidateStructuredOutput()
		fmt.Printf("PROBE compile=%v preflight=%v :: %.70s\n", err, err2, s)
	}
}
