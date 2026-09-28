package ollama

import (
	"encoding/json"
	"github.com/vxcontrol/langchaingo/llms"
	"testing"
)

func TestProbeVrfThink(t *testing.T) {
	for _, m := range []string{"qwen3:8b", "deepseek-r1:8b", "gpt-oss:20b"} {
		for _, e := range []llms.ReasoningEffort{llms.ReasoningLow, llms.ReasoningHigh, llms.ReasoningMax} {
			var o llms.CallOptions
			llms.WithReasoning(e, 0)(&o)
			b, _ := json.Marshal(resolveThink(m, o))
			t.Logf("model=%s effort=%s think=%s", m, e, b)
		}
	}
}
