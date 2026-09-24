package chains

import (
	"context"
	"fmt"
	"maps"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/memory"
	"github.com/vxcontrol/langchaingo/schema"
)

const (
	_mapRerankDocumentsDefaultDocumentTemplate = "{{.page_content}}"
	_mapRerankDocumentsDefaultRankKey          = "score"
	_mapRerankDocumentsDefaultAnswerKey        = "answer"
	_mapRerankAnswerLabel                      = "Helpful Answer:"
)

// _mapRerankScoreRE matches the "Score:" line that ends an answer.
var _mapRerankScoreRE = regexp.MustCompile(`(?m)^[ \t]*Score:[ \t]*(\d*)`)

type MapRerankDocuments struct {
	// Chain used to rerank the documents.
	LLMChain *LLMChain

	// Number of workers to concurrently run apply on documents.
	MaxConcurrentWorkers int

	// The input variable where the documents should be placed int the LLMChain.
	LLMChainInputVariableName string

	// The name of the document variable in the LLMChain.
	DocumentVariableName string

	// Key used to access document inputs.
	InputKey string

	// Key used to access map results.
	OutputKey string

	// Key used for comparison to sort documents.
	RankKey string

	// Key used to return the answer.
	AnswerKey string

	// When true, the intermediate steps of the map rerank are returned.
	ReturnIntermediateSteps bool
}

var _ Chain = MapRerankDocuments{}

// NewMapRerankDocuments creates a new map rerank documents chain.
func NewMapRerankDocuments(mapRerankLLMChain *LLMChain) *MapRerankDocuments {
	mapRerankLLMChain.OutputParser = mapRerankOutputParser{}

	return &MapRerankDocuments{
		LLMChain:                  mapRerankLLMChain,
		MaxConcurrentWorkers:      1,
		LLMChainInputVariableName: _combineDocumentsDefaultDocumentVariableName,
		DocumentVariableName:      _combineDocumentsDefaultDocumentVariableName,
		InputKey:                  _combineDocumentsDefaultInputKey,
		OutputKey:                 _combineDocumentsDefaultOutputKey,
		RankKey:                   _mapRerankDocumentsDefaultRankKey,
		AnswerKey:                 _mapRerankDocumentsDefaultAnswerKey,
	}
}

// Call handles the inner logic of the MapRerankDocuments chain.
func (c MapRerankDocuments) Call(ctx context.Context, values map[string]any, options ...ChainCallOption) (map[string]any, error) { //nolint:lll
	// Get the documents from the input key.
	docs, ok := values[c.InputKey].([]schema.Document)

	if !ok {
		return nil, fmt.Errorf("%w: %w", ErrInvalidInputValues, ErrInputValuesWrongType)
	}

	if len(docs) == 0 {
		return nil, fmt.Errorf("%w: documents slice has no elements", ErrInvalidInputValues)
	}

	applyInputs := c.getApplyInputs(values, docs)
	mapResults, err := Apply(ctx, c.LLMChain, applyInputs, c.MaxConcurrentWorkers, options...)
	if err != nil {
		return nil, err
	}

	// create a slice of outputs to return after sorting the ranks.
	outputs := make([]map[string]any, len(mapResults))

	for i, res := range mapResults {
		rankedAnswer, ok := res[c.LLMChain.OutputKey].(map[string]string)
		if !ok {
			return nil, ErrInvalidOutputValues
		}

		outputs[i] = c.parseMapResults(rankedAnswer)
		if _, ok := outputs[i][c.RankKey]; !ok {
			return nil, fmt.Errorf("%w: the map result has no %q key to rank by", ErrInvalidOutputValues, c.RankKey)
		}
	}

	// A stable sort keeps the document order among equal scores.
	sort.SliceStable(outputs, func(i, j int) bool {
		return c.score(outputs[i]) > c.score(outputs[j])
	})

	return c.formatOutputs(outputs), nil
}

// score returns the rank of a map result. A result without an integer score,
// or with an empty answer, ranks as 0, the score the prompt asks for when the
// context does not answer the question.
func (c MapRerankDocuments) score(output map[string]any) int {
	if answer, ok := output[c.AnswerKey].(string); ok && strings.TrimSpace(answer) == "" {
		return 0
	}
	s, _ := output[c.RankKey].(string)
	score, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0
	}
	return score
}

// getInputVariable returns the input variable name to use for the LLM chain.
func (c MapRerankDocuments) getInputVariable(givenInputName string, chainInputVariables []string) string {
	if len(chainInputVariables) == 1 {
		return chainInputVariables[0]
	}

	return givenInputName
}

// getApplyInputs returns the inputs to use for the apply call.
func (c MapRerankDocuments) getApplyInputs(values map[string]any, docs []schema.Document) []map[string]any {
	llmChainInputVariable := c.getInputVariable(c.LLMChainInputVariableName, c.LLMChain.GetInputKeys())
	inputs := make([]map[string]any, 0, len(docs))
	for _, d := range docs {
		curInput := c.copyInputValuesWithoutInputKey(values)
		curInput[llmChainInputVariable] = d.PageContent
		inputs = append(inputs, curInput)
	}

	return inputs
}

// copyInputValuesWithoutInputKey copies the input values without the input key.
func (c MapRerankDocuments) copyInputValuesWithoutInputKey(inputValues map[string]any) map[string]any {
	inputValuesCopy := make(map[string]any)
	maps.Copy(inputValuesCopy, inputValues)
	delete(inputValuesCopy, c.InputKey)
	return inputValuesCopy
}

// parseMapResults converts the map[string]string results to map[string]any to be usable by chain calls.
func (c MapRerankDocuments) parseMapResults(inputs map[string]string) map[string]any {
	outputs := make(map[string]any)

	for i, input := range inputs {
		outputs[i] = input
	}

	return outputs
}

// formatOutputs returns the first output and the intermediate steps, if enabled.
func (c MapRerankDocuments) formatOutputs(outputs []map[string]any) map[string]any {
	if len(outputs) == 0 {
		return nil
	}

	formattedOutputs := make(map[string]any)
	answerOutput := maps.Clone(outputs[0])

	formattedOutputs[c.LLMChain.OutputKey] = answerOutput[c.AnswerKey]

	if !c.ReturnIntermediateSteps {
		return formattedOutputs
	}

	formattedOutputs[_intermediateStepsOutputKey] = outputs

	return formattedOutputs
}

// GetInputKeys returns the input keys for the MapRerankDocuments chain.
func (c MapRerankDocuments) GetInputKeys() []string {
	inputKeys := []string{c.InputKey}
	for _, key := range c.LLMChain.GetInputKeys() {
		if key == c.DocumentVariableName {
			continue
		}
		inputKeys = append(inputKeys, key)
	}

	return inputKeys
}

// GetOutputKeys returns the output keys for the MapRerankDocuments chain.
func (c MapRerankDocuments) GetOutputKeys() []string {
	outputKeys := c.LLMChain.GetOutputKeys()

	if c.ReturnIntermediateSteps {
		outputKeys = append(outputKeys, _intermediateStepsOutputKey)
	}

	return outputKeys
}

// GetMemory returns the memory for the MapRerankDocuments chain.
func (c MapRerankDocuments) GetMemory() schema.Memory { //nolint:ireturn
	return memory.NewSimple()
}

// mapRerankOutputParser parses the answer and the score of one document. An
// output without a "Score:" line, such as a plain "I don't know.", is kept as
// the answer with an empty score, which ranks as 0, so one such document does
// not fail the whole chain.
type mapRerankOutputParser struct{}

var _ schema.OutputParser[any] = mapRerankOutputParser{}

// GetFormatInstructions returns instructions on the expected output format.
func (p mapRerankOutputParser) GetFormatInstructions() string {
	return "Your output should be an answer followed by a line \"Score: [score between 0 and 100]\"."
}

// Parse parses the output of an LLM into the answer and the score. The answer
// may span lines and ends at the first "Score:" line; whatever precedes the
// prompt's "Helpful Answer:" label, such as the question the model repeated,
// is not part of it.
func (p mapRerankOutputParser) Parse(text string) (any, error) {
	answer, score := text, ""
	if match := _mapRerankScoreRE.FindStringSubmatchIndex(text); match != nil {
		answer, score = text[:match[0]], text[match[2]:match[3]]
	}
	if i := strings.LastIndex(answer, _mapRerankAnswerLabel); i >= 0 {
		answer = answer[i+len(_mapRerankAnswerLabel):]
	}

	return map[string]string{
		_mapRerankDocumentsDefaultAnswerKey: strings.TrimSpace(answer),
		_mapRerankDocumentsDefaultRankKey:   score,
	}, nil
}

// ParseWithPrompt does the same as Parse.
func (p mapRerankOutputParser) ParseWithPrompt(text string, _ llms.PromptValue) (any, error) {
	return p.Parse(text)
}

// Type returns the type of the parser.
func (p mapRerankOutputParser) Type() string {
	return "map_rerank_parser"
}
