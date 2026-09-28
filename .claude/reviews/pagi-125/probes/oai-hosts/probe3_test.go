package openai_test

import (
	"encoding/json"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeQwenOff(t *testing.T) {
	for _, m := range []string{"Qwen/Qwen3-8B", "qwen3:8b", "qwen3:30b-a3b", "Qwen/QwQ-32B", "qwq-32b", "qwen/qwen3-32b", "Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Thinking-2507", "qwen3-coder-plus", "qwen3.5-397b-a17b", "qwen2.5-72b-instruct"} {
		p, _, err := probeCapture2(t, m, nil, llms.WithReasoningDisabled())
		b, _ := json.Marshal(p)
		t.Logf("%s off: err=%v body=%s", m, err, b)
		p, _, err = probeCapture2(t, m, nil, llms.WithReasoning(llms.ReasoningHigh, 0))
		b, _ = json.Marshal(p)
		t.Logf("%s high: err=%v body=%s", m, err, b)
	}
}
