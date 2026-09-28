package llms_test

import (
	"fmt"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestProbeRS(t *testing.T) {
	models := []string{"o3-mini", "o1", "o4-mini", "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5-pro", "gpt-4.1", "gpt-4o", "deepseek-r1", "deepseek-reasoner", "qwq-32b", "qwen3-8b",
		"claude-3-7-sonnet-20250219", "claude-sonnet-4-5", "claude-opus-4-7", "claude-sonnet-4-latest", "anthropic.claude-3-7-sonnet-20250219-v1:0",
		"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-3-pro-preview", "gpt-oss-120b", "openai.gpt-oss-120b-1:0", "amazon.nova-pro-v1:0", "grok-3-mini", "magistral-medium-latest", "llama3.2"}
	provs := []reasoning.Provider{1, 2, 3, 4}
	for _, m := range models {
		for _, p := range provs {
			s := llms.ReasoningSupportFor(m, p)
			d := "nil"
			if s.DefaultOn != nil {
				d = fmt.Sprint(*s.DefaultOn)
			}
			fmt.Printf("RS %-45s p=%d sup=%v known=%v cd=%v rs=%v eff=%v def=%s\n", m, p, s.Supported, s.Known, s.CannotDisable, s.RejectsSampling, s.Efforts, d)
		}
	}
}
