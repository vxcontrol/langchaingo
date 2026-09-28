package googleai

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeV3Gem(t *testing.T) {
	for _, m := range []string{"gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts", "gemini-2.5-flash-image", "gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview", "gemini-2.5-flash-live-translate", "gemini-2.5-flash"} {
		var on, off llms.CallOptions
		llms.WithReasoning(llms.ReasoningHigh, 0)(&on)
		llms.WithReasoningDisabled()(&off)
		c1, e1 := resolveThinkingConfig(m, on.Reasoning, 4096)
		c2, e2 := resolveThinkingConfig(m, off.Reasoning, 4096)
		b1, _ := json.Marshal(c1)
		b2, _ := json.Marshal(c2)
		fmt.Printf("%s on=%s err=%v | off=%s err=%v\n", m, b1, e1, b2, e2)
	}
}
