package chains

import (
	"os"
	"testing"

	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/prompts"
	"github.com/vxcontrol/langchaingo/schema"

	"github.com/stretchr/testify/require"
)

func TestStuffDocuments(t *testing.T) {
	t.Parallel()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	model, err := openai.New()
	require.NoError(t, err)

	prompt := prompts.NewPromptTemplate(
		"Write {{.context}}",
		[]string{"context"},
	)
	require.NoError(t, err)

	llmChain := NewLLMChain(model, prompt)
	chain := NewStuffDocuments(llmChain)

	docs := []schema.Document{
		{PageContent: "foo"},
		{PageContent: "bar"},
		{PageContent: "baz"},
	}

	result, err := Call(t.Context(), chain, map[string]any{
		"input_documents": docs,
	})
	require.NoError(t, err)
	for _, key := range chain.GetOutputKeys() {
		_, ok := result[key]
		require.True(t, ok)
	}
}

func TestStuffDocuments_joinDocs(t *testing.T) {
	t.Parallel()

	testcases := []struct {
		name string
		docs []schema.Document
		want string
	}{
		{
			name: "empty",
			docs: []schema.Document{},
			want: "",
		},
		{
			name: "single",
			docs: []schema.Document{
				{PageContent: "foo"},
			},
			want: "\n<document>\n<content>foo</content>\n\n</document>\n",
		},
		{
			name: "multiple",
			docs: []schema.Document{
				{PageContent: "foo"},
				{PageContent: "bar"},
			},
			want: "\n<document>\n<content>foo</content>\n\n</document>\n\n" +
				"\n\n<document>\n<content>bar</content>\n\n</document>\n",
		},
		{
			name: "multiple with separator",
			docs: []schema.Document{
				{PageContent: "foo"},
				{PageContent: "bar\n\n"},
			},
			want: "\n<document>\n<content>foo</content>\n\n</document>\n\n" +
				"\n\n<document>\n<content>bar\n\n</content>\n\n</document>\n",
		},
	}

	chain := NewStuffDocuments(&LLMChain{})

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := chain.joinDocuments(tc.docs)
			require.Equal(t, tc.want, got)
		})
	}
}
