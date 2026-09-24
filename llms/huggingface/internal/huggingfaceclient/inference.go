package huggingfaceclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

var (
	ErrUnexpectedStatusCode = errors.New("unexpected status code")
	ErrNoChoices            = errors.New("no choices in response")
)

type chatCompletionsPayload struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	Seed        *int          `json:"seed,omitempty"`
	Effort      string        `json:"reasoning_effort,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionsResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

func (c *Client) chatCompletionsURL() string {
	if c.provider == "" {
		return c.url + "/v1/chat/completions"
	}
	return fmt.Sprintf("%s/%s/v1/chat/completions", c.url, c.provider)
}

func (c *Client) runChatCompletions(ctx context.Context, payload *chatCompletionsPayload) (*InferenceResponse, error) {
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.chatCompletionsURL(), bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.Token)
	req.Header.Set("Content-Type", "application/json")

	r, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()

	if r.StatusCode != http.StatusOK {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}

		if len(b) > 0 {
			return nil, fmt.Errorf("%w: %d, body: %s", ErrUnexpectedStatusCode, r.StatusCode, string(b))
		}
		return nil, fmt.Errorf("%w: %d", ErrUnexpectedStatusCode, r.StatusCode)
	}

	var response chatCompletionsResponse
	if err := json.NewDecoder(r.Body).Decode(&response); err != nil {
		return nil, err
	}
	if len(response.Choices) == 0 {
		return nil, ErrNoChoices
	}

	return &InferenceResponse{
		Text:       response.Choices[0].Message.Content,
		StopReason: response.Choices[0].FinishReason,
	}, nil
}
