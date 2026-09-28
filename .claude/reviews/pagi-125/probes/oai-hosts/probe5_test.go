package openai_test

import (
	"encoding/json"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeFunctionsRule(t *testing.T) {
	fn := []llms.FunctionDefinition{{Name: "f", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}
	tool := []llms.Tool{{Type: "function", Function: &fn[0]}}
	for _, m := range []string{"gpt-5.6", "gpt-5.4"} {
		p, _, err := probeCapture2(t, m, nil, llms.WithFunctions(fn))
		b, _ := json.Marshal(p)
		t.Logf("%s functions default: err=%v body=%s", m, err, b)
		p, _, err = probeCapture2(t, m, nil, llms.WithTools(tool))
		b, _ = json.Marshal(p)
		t.Logf("%s tools default: err=%v body=%s", m, err, b)
		p, _, err = probeCapture2(t, m, nil, llms.WithFunctions(fn), llms.WithReasoning(llms.ReasoningHigh, 0))
		b, _ = json.Marshal(p)
		t.Logf("%s functions high: err=%v body=%s", m, err, b)
		p, _, err = probeCapture2(t, m, nil, llms.WithTools(tool), llms.WithReasoning(llms.ReasoningHigh, 0))
		b, _ = json.Marshal(p)
		t.Logf("%s tools high: err=%v body=%s", m, err, b)
	}
}
