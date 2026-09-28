package chains

import (
	"fmt"
	"slices"
	"testing"

	"github.com/vxcontrol/langchaingo/prompts"
	"github.com/vxcontrol/langchaingo/schema"

	"github.com/stretchr/testify/require"
)

func TestMapRerankInputVariables(t *testing.T) {
	t.Parallel()

	mapRerankLLMChain := NewLLMChain(
		&testLanguageModel{},
		prompts.NewPromptTemplate("{{.text}} {{.foo}}", []string{"text", "foo"}),
	)

	c := MapRerankDocuments{
		LLMChain:                  mapRerankLLMChain,
		DocumentVariableName:      "texts",
		LLMChainInputVariableName: "text",
		InputKey:                  "input",
	}

	inputKeys := c.GetInputKeys()
	expectedLength := 3
	require.Len(t, inputKeys, expectedLength)
}

func TestMapRerankDocumentsCall(t *testing.T) {
	ctx := t.Context()
	t.Parallel()

	mapRerankLLMChain := NewLLMChain(
		&testLanguageModel{},
		prompts.NewPromptTemplate("{{.context}}", []string{"context"}),
	)

	docs := []schema.Document{
		{PageContent: "Test Low\nScore: 20"},
		{PageContent: "Test High\nScore: 100"},
	}

	mapRerankDocumentsChain := NewMapRerankDocuments(mapRerankLLMChain)

	// Test that the answer is the highest scoring document.
	answer, err := Run(ctx, mapRerankDocumentsChain, docs)

	require.NoError(t, err)
	require.Equal(t, "Test High", answer)

	// Test that the answer cannot be processed if ReturnIntermediateSteps is true.
	mapRerankDocumentsChain.ReturnIntermediateSteps = true
	_, err = Run(ctx, mapRerankDocumentsChain, docs)

	require.Error(t, err)

	// Test that scores that cannot be processed rank as 0 and keep the document order.
	mapRerankDocumentsChain.ReturnIntermediateSteps = false
	docs = []schema.Document{
		{PageContent: "Test Low\nScore:"},
		{PageContent: "Test High\nScore:"},
	}

	answer, err = Run(ctx, mapRerankDocumentsChain, docs)

	require.NoError(t, err)
	require.Equal(t, "Test Low", answer)
}

func TestMapRerankOutputParser(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		text   string
		answer string
		score  string
	}{
		{name: "answer and score", text: "Test High\nScore: 100", answer: "Test High", score: "100"},
		{name: "blank line before score", text: "The lion is Leo.\n\nScore: 100", answer: "The lion is Leo.", score: "100"},
		{name: "multiline answer", text: "Leo is a lion.\nHe is brave.\nScore: 90", answer: "Leo is a lion.\nHe is brave.", score: "90"},
		{name: "helpful answer label", text: "Helpful Answer: Leo\nScore: 100", answer: "Leo", score: "100"},
		{name: "repeated question", text: "Question: Who?\nHelpful Answer: Leo\nScore: 100", answer: "Leo", score: "100"},
		{name: "padded score", text: "  Leo \nScore:  85 \n", answer: "Leo", score: "85"},
		{name: "carriage return", text: "Leo\r\nScore: 70", answer: "Leo", score: "70"},
		{name: "further example", text: "Leo\nScore: 100\nQuestion: Why?\nHelpful Answer: No\nScore: 0", answer: "Leo", score: "100"},
		{name: "score is not a number", text: "Leo\nScore: N/A", answer: "Leo", score: ""},
		{name: "no score line", text: "I don't know.", answer: "I don't know.", score: ""},
		{name: "no score line with label", text: "Question: Who?\nHelpful Answer: I don't know.\n", answer: "I don't know.", score: ""},
		{name: "score inside a line", text: "The final Score: 3\nScore: 80", answer: "The final Score: 3", score: "80"},
		{name: "last label wins", text: "Helpful Answer: a\nHelpful Answer: b\nScore: 50", answer: "b", score: "50"},
		{name: "score before the answer", text: "Score: 100\nHelpful Answer: Leo", answer: "", score: "100"},
	}

	parser := mapRerankOutputParser{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			parsed, err := parser.ParseWithPrompt(tt.text, nil)
			require.NoError(t, err)
			require.Equal(t, map[string]string{"answer": tt.answer, "score": tt.score}, parsed)
		})
	}
}

func TestMapRerankDocumentsUnscoredAnswerRanksZero(t *testing.T) {
	t.Parallel()
	ctx := t.Context()

	mapRerankLLMChain := NewLLMChain(
		&testLanguageModel{},
		prompts.NewPromptTemplate("{{.context}}", []string{"context"}),
	)
	mapRerankDocumentsChain := NewMapRerankDocuments(mapRerankLLMChain)
	mapRerankDocumentsChain.ReturnIntermediateSteps = true

	docs := []schema.Document{
		{PageContent: "I don't know."},
		{PageContent: "Other\nScore: 50"},
		{PageContent: "Leo\nScore: 100 "},
	}

	result, err := Call(ctx, mapRerankDocumentsChain, map[string]any{"input_documents": docs})
	require.NoError(t, err)
	require.Equal(t, "Leo", result["text"])
	require.Equal(t, []map[string]any{
		{"answer": "Leo", "score": "100"},
		{"answer": "Other", "score": "50"},
		{"answer": "I don't know.", "score": ""},
	}, result[_intermediateStepsOutputKey])
}

func TestMapRerankDocumentsStableRanking(t *testing.T) {
	t.Parallel()
	ctx := t.Context()

	mapRerankLLMChain := NewLLMChain(
		&testLanguageModel{},
		prompts.NewPromptTemplate("{{.context}}", []string{"context"}),
	)
	mapRerankDocumentsChain := NewMapRerankDocuments(mapRerankLLMChain)
	mapRerankDocumentsChain.ReturnIntermediateSteps = true

	// Enough documents for the sort to leave insertion sort, with repeated scores.
	scores := []string{"50", "100", "0", "100", "50", "0"}
	docs := make([]schema.Document, 0, 60)
	byScore := map[string][]map[string]any{}
	for i := range 60 {
		score := scores[i%len(scores)]
		answer := fmt.Sprintf("answer %02d", i)
		docs = append(docs, schema.Document{PageContent: answer + "\nScore: " + score})
		byScore[score] = append(byScore[score], map[string]any{"answer": answer, "score": score})
	}
	expected := slices.Concat(byScore["100"], byScore["50"], byScore["0"])

	result, err := Call(ctx, mapRerankDocumentsChain, map[string]any{"input_documents": docs})
	require.NoError(t, err)
	require.Equal(t, "answer 01", result["text"])
	require.Equal(t, expected, result[_intermediateStepsOutputKey])
}

func TestMapRerankDocumentsScore(t *testing.T) {
	t.Parallel()

	c := MapRerankDocuments{RankKey: "score", AnswerKey: "answer"}
	require.Equal(t, 42, c.score(map[string]any{"score": " 42 "}))
	require.Equal(t, 42, c.score(map[string]any{"answer": "Leo", "score": "42"}))
	require.Equal(t, 0, c.score(map[string]any{"answer": " ", "score": "100"}))
	require.Equal(t, 0, c.score(map[string]any{"score": "high"}))
	require.Equal(t, 0, c.score(map[string]any{"score": 42}))
	require.Equal(t, 0, c.score(map[string]any{}))
}

func TestMapRerankDocumentsRequiresTheRankKey(t *testing.T) {
	t.Parallel()

	mapRerankLLMChain := NewLLMChain(
		&testLanguageModel{},
		prompts.NewPromptTemplate("{{.context}}", []string{"context"}),
	)
	mapRerankDocumentsChain := NewMapRerankDocuments(mapRerankLLMChain)
	mapRerankDocumentsChain.RankKey = "confidence"

	docs := []schema.Document{
		{PageContent: "Low\nScore: 10"},
		{PageContent: "High\nScore: 100"},
	}

	_, err := Call(t.Context(), mapRerankDocumentsChain, map[string]any{"input_documents": docs})
	require.ErrorIs(t, err, ErrInvalidOutputValues)
}
