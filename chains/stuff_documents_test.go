package chains

import (
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/prompts"
	"github.com/vxcontrol/langchaingo/schema"

	"github.com/stretchr/testify/require"
)

func TestStuffDocuments(t *testing.T) {
	ctx := t.Context()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	// Only run tests in parallel when not recording
	if rr.Replaying() {
		t.Parallel()
	}

	opts := []openai.Option{
		openai.WithHTTPClient(rr.Client()),
	}

	// Only add fake token when NOT recording (i.e., during replay)
	if rr.Replaying() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}
	// When recording, openai.New() will read OPENAI_API_KEY from environment

	model, err := openai.New(opts...)
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

	result, err := Call(ctx, chain, map[string]any{
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
