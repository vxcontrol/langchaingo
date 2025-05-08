package chains

import (
	"os"
	"testing"

	"github.com/vxcontrol/langchaingo/llms/openai"

	"github.com/stretchr/testify/require"
)

func TestLLMMath(t *testing.T) {
	t.Parallel()
	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	llm, err := openai.New()
	require.NoError(t, err)

	chain := NewLLMMathChain(llm)
	q := "what is forty plus three? take that then multiply it by ten thousand divided by 7324.3"
	result, err := Run(t.Context(), chain, q)
	require.NoError(t, err)
	require.Contains(t, result, "58.708", "expected 58.708 in result")
}
