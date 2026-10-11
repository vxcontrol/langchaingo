package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestDashScopeGuestsAreSentTheEffortsModelStudioTakes(t *testing.T) {
	t.Parallel()

	const dashScope = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	for _, tc := range []struct {
		model string
		asked llms.ReasoningEffort
		sent  string
	}{
		{"kimi/kimi-k3", llms.ReasoningHigh, "max"},
		{"kimi/kimi-k3", llms.ReasoningLow, "max"},
		{"kimi/kimi-k4", llms.ReasoningHigh, "max"},
		{"deepseek-v4-pro", llms.ReasoningMinimal, "low"},
		{"deepseek-v4.1-flash", llms.ReasoningMinimal, "low"},
		{"deepseek-v4-pro", llms.ReasoningHigh, "high"},
		{"kimi-k3", llms.ReasoningLow, "low"},
		{"glm-5.1", llms.ReasoningMax, "xhigh"},
		{"glm-5.1", llms.ReasoningXHigh, "xhigh"},
	} {
		for _, baseURL := range []string{dashScope, "https://maas.qwencloudapi.com/compatible-mode/v1"} {
			body, warnings := hostCall(t, baseURL, tc.model, llms.WithReasoning(tc.asked, 0))
			assert.Equal(t, tc.sent, body["reasoning_effort"], "%s %s %s", baseURL, tc.model, tc.asked)
			if string(tc.asked) != tc.sent {
				assert.Contains(t, warnings, "WithReasoning", "%s %s %s", baseURL, tc.model, tc.asked)
			} else {
				assert.NotContains(t, warnings, "WithReasoning", "%s %s %s", baseURL, tc.model, tc.asked)
			}
		}
	}

	body, _ := hostCall(t, "http://litellm.internal/v1", "deepseek-v4-pro", llms.WithReasoning(llms.ReasoningMinimal, 0))
	assert.Equal(t, "minimal", body["reasoning_effort"], "another host keeps its own effort set")
}
