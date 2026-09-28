package ollama

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestProbeGptOss(t *testing.T) {
	efforts := []llms.ReasoningEffort{"", "minimal", "low", "medium", "high", "xhigh", "max"}
	for _, e := range efforts {
		t.Logf("effort=%q door=%q shared=%q", e, gptOSSLevel(e), reasoning.GptOssEffort(string(e)))
	}
	cfgs := []*llms.ReasoningConfig{
		{Mode: llms.ReasoningOn},
		{Adaptive: true, Tokens: 100},
		{Tokens: 100}, {Tokens: 100000},
		{Effort: "minimal"}, {Effort: "xhigh"}, {Effort: "max"},
	}
	for _, c := range cfgs {
		opts := llms.CallOptions{Reasoning: c}
		th := resolveThink("gpt-oss:20b", opts)
		got := "<nil>"
		if th != nil {
			got = th.String()
		}
		t.Logf("cfg=%+v resolved effort=%q think=%s shared=%q", *c, c.GetEffort(opts.GetMaxTokens()), got, reasoning.GptOssEffort(string(c.GetEffort(opts.GetMaxTokens()))))
	}
	for _, m := range []string{"gpt-oss:20b", "gpt-oss", "GPT-OSS:120b", "hf.co/unsloth/gpt-oss-20b-GGUF", "openai.gpt-oss-20b", "library/gpt-oss:latest", "gpt-oss-safeguard:20b"} {
		t.Logf("model=%q door=%v shared=%v", m, takesOnlyGPTOSSLevels(m), reasoning.OllamaEffortsFor(m) != nil)
	}
}
