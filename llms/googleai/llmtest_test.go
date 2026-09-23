package googleai

import (
	"context"
	"os"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/testing/llmtest"
)

func TestLLM(t *testing.T) {
	if os.Getenv("GOOGLE_API_KEY") == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}

	ctx := context.Background()
	llm, err := New(ctx,
		WithAPIKey(os.Getenv("GOOGLE_API_KEY")),
		WithDefaultModel("gemini-3.8-flash"),
	)
	if err != nil {
		t.Fatalf("Failed to create Google AI LLM: %v", err)
	}

	// gemini-3.8-flash cannot stop thinking: it refuses the minimal level and
	// reasoning off, and its thoughts count against the output budget. At its
	// default level it spends up to about 160 tokens on thoughts, so the suite's
	// 10- and 50-token answers end on MAX_TOKENS with no text unless the door
	// declares room to think first.
	llmtest.TestLLM(t, llm, llmtest.WithCallOptions(llms.WithMaxTokens(512)))
}
