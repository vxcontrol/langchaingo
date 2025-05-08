package chains

import (
	"os"
	"testing"

	"github.com/vxcontrol/langchaingo/documentloaders"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/textsplitter"

	"github.com/stretchr/testify/require"
)

func loadTestData(t *testing.T) []schema.Document {
	t.Helper()

	file, err := os.Open("./testdata/mouse_story.txt")
	require.NoError(t, err)

	docs, err := documentloaders.NewText(file).LoadAndSplit(
		t.Context(),
		textsplitter.NewRecursiveCharacter(),
	)
	require.NoError(t, err)

	return docs
}

func TestStuffSummarization(t *testing.T) {
	t.Parallel()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	llm, err := openai.New()
	require.NoError(t, err)

	docs := loadTestData(t)

	chain := LoadStuffSummarization(llm)
	_, err = Call(
		t.Context(),
		chain,
		map[string]any{"input_documents": docs},
	)
	require.NoError(t, err)
}

func TestRefineSummarization(t *testing.T) {
	t.Parallel()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	llm, err := openai.New()
	require.NoError(t, err)

	docs := loadTestData(t)

	chain := LoadRefineSummarization(llm)
	_, err = Call(
		t.Context(),
		chain,
		map[string]any{"input_documents": docs},
	)
	require.NoError(t, err)
}

func TestMapReduceSummarization(t *testing.T) {
	t.Parallel()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	llm, err := openai.New()
	require.NoError(t, err)

	docs := loadTestData(t)

	chain := LoadMapReduceSummarization(llm)
	_, err = Run(
		t.Context(),
		chain,
		docs,
	)
	require.NoError(t, err)
}
