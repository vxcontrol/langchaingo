package reasoning

import "testing"

func TestProbeSkeptic(t *testing.T) {
	for _, m := range []string{"deepseek-r1:8b", "deepseek-ai/DeepSeek-V3", "z-ai/glm-4.6", "glm4:9b", "zai-org/GLM-4.5-Air"} {
		t.Logf("%s TakesNoJSONSchema=%v", m, TakesNoJSONSchema(m))
	}
	for _, m := range []string{"qwen3-8b", "qwen/qwen3-8b", "qwen3:8b", "Qwen/Qwen3-8B"} {
		t.Logf("%s RequiresStream=%v", m, QwenThinkingRequiresStream(m))
	}
}
