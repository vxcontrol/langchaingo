package chains

import (
	"testing"

	"github.com/vxcontrol/langchaingo/prompts"
	"github.com/vxcontrol/langchaingo/schema"

	"github.com/stretchr/testify/require"
)

func TestMapReduceInputVariables(t *testing.T) {
	t.Parallel()

	c := MapReduceDocuments{
		LLMChain: NewLLMChain(
			&testLanguageModel{},
			prompts.NewPromptTemplate("{{.text}} {{.foo}}", []string{"text", "foo"}),
		),
		ReduceChain: NewLLMChain(
			&testLanguageModel{},
			prompts.NewPromptTemplate("{{.texts}} {{.baz}}", []string{"texts", "baz"}),
		),
		ReduceDocumentVariableName: "texts",
		LLMChainInputVariableName:  "text",
		InputKey:                   "input",
	}

	inputKeys := c.GetInputKeys()
	expectedLength := 3
	require.Len(t, inputKeys, expectedLength)
}

func TestMapReduce(t *testing.T) {
	t.Parallel()
	ctx := t.Context()

	c := NewMapReduceDocuments(
		NewLLMChain(
			&testLanguageModel{},
			prompts.NewPromptTemplate("{{.context}}", []string{"context"}),
		),
		NewStuffDocuments(
			NewLLMChain(
				&testLanguageModel{},
				prompts.NewPromptTemplate("{{.context}}", []string{"context"}),
			),
		),
	)

	result, err := Run(ctx, c, []schema.Document{
		{PageContent: "foo"},
		{PageContent: "boo"},
		{PageContent: "zoo"},
		{PageContent: "doo"},
	})
	require.NoError(t, err)
	require.Equal(t, "<document>\n"+
		"<content>foo</content>\n\n</document>\n\n\n\n<document>\n<content>boo</content>\n\n"+
		"</document>\n\n"+
		"\n\n<document>\n"+
		"<content>zoo</content>\n\n</document>\n\n\n\n<document>\n<content>doo</content>\n\n"+
		"</document>", result)
}
