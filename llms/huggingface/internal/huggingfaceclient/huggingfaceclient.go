package huggingfaceclient

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/vxcontrol/langchaingo/httputil"
)

var (
	ErrInvalidToken  = errors.New("invalid token")
	ErrEmptyResponse = errors.New("empty response")
)

type Client struct {
	Token      string
	Model      string
	url        string
	httpClient *http.Client
	provider   string // Inference provider for router-based requests
}

func New(token, model, url string, opts ...Option) (*Client, error) {
	if token == "" {
		return nil, ErrInvalidToken
	}

	client := &Client{
		Token:      token,
		Model:      model,
		url:        url,
		httpClient: httputil.DefaultClient,
	}

	for _, opt := range opts {
		opt(client)
	}

	return client, nil
}

// Option configures a HuggingFace client.
type Option func(*Client)

// WithHTTPClient sets a custom HTTP client for the HuggingFace client.
func WithHTTPClient(httpClient *http.Client) Option {
	return func(c *Client) {
		c.httpClient = httpClient
	}
}

// WithProvider sets the inference provider for router-based requests.
func WithProvider(provider string) Option {
	return func(c *Client) {
		c.provider = provider
	}
}

type InferenceRequest struct {
	Model       string
	Prompt      string
	Temperature *float64
	TopP        *float64
	MaxTokens   *int
	Seed        *int
	Effort      string
}

type InferenceResponse struct {
	Text string
	// StopReason is the vendor's finish reason, empty when the door did not report one.
	StopReason string
}

func (c *Client) RunInference(ctx context.Context, request *InferenceRequest) (*InferenceResponse, error) {
	payload := &chatCompletionsPayload{
		Model:       request.Model,
		Messages:    []chatMessage{{Role: "user", Content: request.Prompt}},
		Effort:      request.Effort,
		Temperature: request.Temperature,
		TopP:        request.TopP,
		MaxTokens:   request.MaxTokens,
		Seed:        request.Seed,
	}

	resp, err := c.runChatCompletions(ctx, payload)
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}
	return resp, nil
}

// EmbeddingRequest is a request to create an embedding.
type EmbeddingRequest struct {
	Inputs []string `json:"inputs"`
}

// CreateEmbedding creates embeddings.
func (c *Client) CreateEmbedding(
	ctx context.Context,
	model string,
	task string,
	r *EmbeddingRequest,
) ([][]float32, error) {
	resp, err := c.createEmbedding(ctx, model, task, &embeddingPayload{
		Inputs: r.Inputs,
	})
	if err != nil {
		return nil, err
	}

	if len(resp) == 0 {
		return nil, ErrEmptyResponse
	}

	return resp, nil
}
