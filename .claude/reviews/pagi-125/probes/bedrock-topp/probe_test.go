package bedrockclient

import (
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeConverseTopPFloor(t *testing.T) {
	for _, topP := range []float64{0.95, 0.96, 0.97} {
		p := topP
		mt := 4096
		built, err := NewConverseClient(nil).buildConverseInput(&ConverseInput{
			ModelID:         "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
			Messages:        []Message{{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"}},
			MaxTokens:       &mt,
			TopP:            &p,
			ReasoningConfig: &llms.ReasoningConfig{Tokens: 1024},
		})
		var gotP, gotT any = nil, nil
		if built.InferenceConfig.TopP != nil {
			gotP = *built.InferenceConfig.TopP
		}
		if built.InferenceConfig.Temperature != nil {
			gotT = *built.InferenceConfig.Temperature
		}
		t.Logf("converse topP asked=%v err=%v sent top_p=%v temperature=%v", topP, err, gotP, gotT)

		in := anthropicTextGenerationInput{MaxTokens: 4096, TopP: topP}
		_ = applyAnthropicReasoning(&in, &llms.ReasoningConfig{Tokens: 1024}, "us.anthropic.claude-sonnet-4-5-20250929-v1:0", 4096)
		t.Logf("legacy   topP asked=%v sent top_p=%v temperature=%v", topP, in.TopP, in.Temperature)
	}
}
