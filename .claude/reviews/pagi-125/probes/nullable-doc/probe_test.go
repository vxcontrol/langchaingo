package structuredoutput_test

import (
	"encoding/json"
	"testing"

	"github.com/vxcontrol/langchaingo/llms/structuredoutput"
)

func TestProbeNullableDoc(t *testing.T) {
	c, err := structuredoutput.Compile(json.RawMessage(`{"type":"object","properties":{"a":{"type":"string","nullable":true}},"required":["a"]}`))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("validate {\"a\":null}: err=%v", c.ValidateText(`{"a":null}`))
}
