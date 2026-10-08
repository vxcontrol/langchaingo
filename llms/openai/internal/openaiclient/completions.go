package openaiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

// CompletionRequest is a request to complete a completion.
type CompletionRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Temperature float64 `json:"temperature,omitempty"`
	// Deprecated: Use MaxCompletionTokens
	MaxTokens           int      `json:"-"`
	MaxCompletionTokens int      `json:"max_completion_tokens,omitempty"`
	N                   int      `json:"n,omitempty"`
	FrequencyPenalty    float64  `json:"frequency_penalty,omitempty"`
	PresencePenalty     float64  `json:"presence_penalty,omitempty"`
	TopP                float64  `json:"top_p,omitempty"`
	StopWords           []string `json:"stop,omitempty"`
	Seed                int      `json:"seed,omitempty"`

	// Reasoning effort is enabling reasoning if the model supports it.
	ReasoningEffort *llms.ReasoningEffort `json:"reasoning_effort,omitempty"`

	// StreamingFunc is a function to be called for each chunk of a streaming response.
	// Return an error to stop streaming early.
	StreamingFunc streaming.Callback `json:"-"`
}

type CompletionResponse struct {
	ID      string  `json:"id,omitempty"`
	Created float64 `json:"created,omitempty"`
	Choices []struct {
		FinishReason string      `json:"finish_reason,omitempty"`
		Index        float64     `json:"index,omitempty"`
		Logprobs     interface{} `json:"logprobs,omitempty"`
		Text         string      `json:"text,omitempty"`
	} `json:"choices,omitempty"`
	Model  string `json:"model,omitempty"`
	Object string `json:"object,omitempty"`
	Usage  struct {
		CompletionTokens float64 `json:"completion_tokens,omitempty"`
		PromptTokens     float64 `json:"prompt_tokens,omitempty"`
		TotalTokens      float64 `json:"total_tokens,omitempty"`
	} `json:"usage,omitempty"`
}

const maxErrorBodyBytes = 64 << 10

type StatusError struct {
	StatusCode int
	Message    string
}

func (e *StatusError) Error() string {
	msg := fmt.Sprintf("API returned unexpected status code: %d", e.StatusCode)
	if e.Message == "" {
		return msg
	}
	return msg + ": " + e.Message
}

func statusError(statusCode int, body io.Reader) error {
	raw, err := io.ReadAll(io.LimitReader(body, maxErrorBodyBytes))
	if err != nil {
		return &StatusError{StatusCode: statusCode}
	}
	raw = bytes.TrimSpace(raw)

	var errResp struct {
		Error *providerError `json:"error"`
	}
	if err := json.Unmarshal(raw, &errResp); err == nil && errResp.Error != nil && errResp.Error.Message != "" {
		return &StatusError{StatusCode: statusCode, Message: errResp.Error.Message}
	}
	return &StatusError{StatusCode: statusCode, Message: string(raw)}
}

func (c *Client) setCompletionDefaults(payload *CompletionRequest) {
	// Set defaults
	if payload.MaxTokens == 0 {
		payload.MaxTokens = 2048
	}

	if len(payload.StopWords) == 0 {
		payload.StopWords = nil
	}

	switch {
	// Prefer the model specified in the payload.
	case payload.Model != "":

	// If no model is set in the payload, take the one specified in the client.
	case c.Model != "":
		payload.Model = c.Model
	// Fallback: use the default model
	default:
		payload.Model = DefaultChatModel
	}
}

// nolint:lll
func (c *Client) createCompletion(ctx context.Context, payload *CompletionRequest) (*ChatCompletionResponse, error) {
	c.setCompletionDefaults(payload)
	return c.createChat(ctx, &ChatRequest{
		Model: payload.Model,
		Messages: []*ChatMessage{
			{Role: "user", Content: payload.Prompt},
		},
		Temperature:         &payload.Temperature,
		TopP:                &payload.TopP,
		MaxCompletionTokens: &payload.MaxTokens,
		N:                   &payload.N,
		StopWords:           payload.StopWords,
		FrequencyPenalty:    &payload.FrequencyPenalty,
		PresencePenalty:     &payload.PresencePenalty,
		StreamingFunc:       payload.StreamingFunc,
		Seed:                &payload.Seed,
		ReasoningEffort:     payload.ReasoningEffort,
	})
}
