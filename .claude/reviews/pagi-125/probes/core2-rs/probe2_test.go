package llms_test

import (
	"fmt"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestProbeRS2(t *testing.T) {
	cases := []struct {
		m string
		p reasoning.Provider
	}{
		{"gpt-oss:20b", 5}, {"gpt-oss:120b-cloud", 5}, {"gpt-oss", 5}, {"qwen3:8b", 5}, {"deepseek-r1:8b", 5}, {"llama3.2", 5}, {"magistral:24b", 5},
		{"us.openai.gpt-oss-120b-1:0", 2}, {"openai.gpt-oss-20b-1:0", 2}, {"amazon.nova-2-lite-v1:0", 2}, {"us.amazon.nova-2-lite-v1:0", 2}, {"global.amazon.nova-2-lite-v1:0", 2},
		{"deepseek.r1-v1:0", 2}, {"us.deepseek.r1-v1:0", 2}, {"global.anthropic.claude-sonnet-4-5-20250929-v1:0", 2}, {"us.anthropic.claude-opus-4-7", 2},
		{"meta.llama3-70b-instruct-v1:0", 2}, {"mistral.mistral-large-2402-v1:0", 2}, {"qwen.qwen3-32b-v1:0", 2}, {"amazon.nova-pro-v1:0", 2}, {"amazon.nova-premier-v1:0", 2},
		{"gemini-2.5-flash-lite", 4}, {"gemini-3-flash-preview", 4}, {"gemma-4-27b-it", 4}, {"models/gemini-2.5-flash", 4}, {"gemini-2.0-flash-thinking-exp", 4},
	}
	for _, c := range cases {
		s := llms.ReasoningSupportFor(c.m, c.p)
		d := "nil"
		if s.DefaultOn != nil {
			d = fmt.Sprint(*s.DefaultOn)
		}
		fmt.Printf("RS2 %-50s p=%d sup=%v known=%v cd=%v rs=%v eff=%v mech=%v def=%s\n", c.m, c.p, s.Supported, s.Known, s.CannotDisable, s.RejectsSampling, s.Efforts, s.Mechanism, d)
	}
}
