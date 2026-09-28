package llmtest_test

import (
	"context"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/testing/llmtest"
)

func TestProbeMockStream(t *testing.T) {
	m := &llmtest.MockLLM{}
	ch, err := m.GenerateContentStream(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	if err != nil {
		t.Fatal(err)
	}
	for c := range ch {
		t.Logf("chunk %q", c.Choices[0].Content)
	}
}
