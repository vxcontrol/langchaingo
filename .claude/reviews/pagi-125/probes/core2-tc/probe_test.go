package openai

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeToolChoiceShapes(t *testing.T) {
	for _, c := range []any{
		map[string]any{"type": "function", "function": map[string]string{"name": "calc"}},
		map[string]any{"type": "function", "function": llms.FunctionReference{Name: "calc"}},
		map[string]any{"type": "function", "function": &llms.FunctionReference{Name: "calc"}},
		map[string]any{"type": "function", "function": map[string]any{"name": "calc"}},
		map[string]string{"type": "function", "name": "calc"},
	} {
		kind, name := llms.ClassifyToolChoice(c)
		in, _ := json.Marshal(c)
		out, _ := json.Marshal(openaiToolChoice(c))
		fmt.Printf("PROBE kind=%v name=%q caller=%s wire=%s\n", kind, name, in, out)
	}
}
