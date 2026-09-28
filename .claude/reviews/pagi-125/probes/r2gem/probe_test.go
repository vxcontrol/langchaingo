package googleai

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeR2Gem(t *testing.T) {
	for _, m := range []string{"gemini-3.1-flash-tts-preview", "gemini-2.5-flash-image", "gemini-3.1-flash-image"} {
		on, errOn := resolveThinkingConfig(m, &llms.ReasoningConfig{Effort: llms.ReasoningHigh}, 1024)
		off, errOff := resolveThinkingConfig(m, &llms.ReasoningConfig{Mode: llms.ReasoningOff}, 1024)
		a, _ := json.Marshal(on)
		b, _ := json.Marshal(off)
		fmt.Printf("GEM %-30s on=%s err=%v | off=%s err=%v\n", m, a, errOn, b, errOff)
	}
}
