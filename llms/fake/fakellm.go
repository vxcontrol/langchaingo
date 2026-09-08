package fake

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type LLM struct {
	mu        sync.Mutex
	responses []string
	index     int
}

func NewFakeLLM(responses []string) *LLM {
	return &LLM{
		responses: responses,
		index:     0,
	}
}

// GenerateContent generate fake content.
func (f *LLM) GenerateContent(ctx context.Context, _ []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	f.mu.Lock()
	if len(f.responses) == 0 {
		f.mu.Unlock()
		return nil, errors.New("no responses configured")
	}
	if f.index >= len(f.responses) {
		f.index = 0 // reset index
	}
	response := f.responses[f.index]
	f.index++
	f.mu.Unlock()

	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}
	if opts.StreamingFunc != nil {
		for _, part := range strings.SplitAfter(response, " ") {
			if err := streaming.CallWithText(ctx, opts.StreamingFunc, part); err != nil {
				return nil, err
			}
		}
		if err := streaming.CallWithDone(ctx, opts.StreamingFunc); err != nil {
			return nil, err
		}
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{Content: response}},
	}, nil
}

// Call  the model with a prompt.
func (f *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	resp, err := f.GenerateContent(ctx, []llms.MessageContent{{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: prompt}}}}, options...)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) < 1 {
		return "", errors.New("empty response from model")
	}
	return resp.Choices[0].Content, nil
}

// Reset the index to 0.
func (f *LLM) Reset() {
	f.mu.Lock()
	f.index = 0
	f.mu.Unlock()
}

// AddResponse adds a response to the list of responses.
func (f *LLM) AddResponse(response string) {
	f.mu.Lock()
	f.responses = append(f.responses, response)
	f.mu.Unlock()
}
