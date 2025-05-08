package chains

import (
	"os"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/callbacks"
	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/prompts"

	"github.com/stretchr/testify/require"
)

func TestLLMChain(t *testing.T) {
	t.Parallel()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	model, err := openai.New()
	require.NoError(t, err)
	model.CallbacksHandler = callbacks.LogHandler{}

	prompt := prompts.NewPromptTemplate(
		"What is the capital of {{.country}}",
		[]string{"country"},
	)
	require.NoError(t, err)

	chain := NewLLMChain(model, prompt)

	result, err := Predict(t.Context(), chain,
		map[string]any{
			"country": "France",
		},
	)
	require.NoError(t, err)
	require.True(t, strings.Contains(result, "Paris"))
}

func TestLLMChainWithChatPromptTemplate(t *testing.T) {
	t.Parallel()

	c := NewLLMChain(
		&testLanguageModel{},
		prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
			prompts.NewAIMessagePromptTemplate("{{.foo}}", []string{"foo"}),
			prompts.NewHumanMessagePromptTemplate("{{.boo}}", []string{"boo"}),
		}),
	)
	result, err := Predict(t.Context(), c, map[string]any{
		"foo": "foo",
		"boo": "boo",
	})
	require.NoError(t, err)
	require.Equal(t, "AI: foo\nHuman: boo", result)
}

func TestLLMChainWithGoogleAI(t *testing.T) {
	t.Parallel()

	genaiKey := os.Getenv("GENAI_API_KEY")
	if genaiKey == "" {
		t.Skip("GENAI_API_KEY not set")
	}

	model, err := googleai.New(t.Context(), googleai.WithAPIKey(genaiKey))
	require.NoError(t, err)
	require.NoError(t, err)
	model.CallbacksHandler = callbacks.LogHandler{}

	prompt := prompts.NewPromptTemplate(
		"What is the capital of {{.country}}",
		[]string{"country"},
	)
	require.NoError(t, err)

	chain := NewLLMChain(model, prompt)

	// chains tramples over defaults for options, so setting these options
	// explicitly is required until https://github.com/tmc/langchaingo/issues/626
	// is fully resolved.
	result, err := Predict(t.Context(), chain,
		map[string]any{
			"country": "France",
		},
		WithCallback(callbacks.LogHandler{}),
	)
	require.NoError(t, err)
	require.True(t, strings.Contains(result, "Paris"))
}
