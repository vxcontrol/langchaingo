package huggingfaceclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type embeddingPayload struct {
	Inputs []string `json:"inputs"`
}

const defaultEmbeddingProvider = "hf-inference"

func (c *Client) embeddingProvider() string {
	if c.provider == "" {
		return defaultEmbeddingProvider
	}
	return c.provider
}

// nolint:lll
func (c *Client) createEmbedding(ctx context.Context, model, task string, payload *embeddingPayload) ([][]float32, error) {
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}
	url := fmt.Sprintf("%s/%s/models/%s/pipeline/%s", c.url, c.embeddingProvider(), model, task)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.Token)
	req.Header.Set("Content-Type", "application/json")

	r, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()

	if r.StatusCode != http.StatusOK {
		var body []byte
		if r.Body != nil {
			body, _ = io.ReadAll(r.Body)
		}
		msg := fmt.Sprintf("API returned unexpected status code: %d for URL: %s", r.StatusCode, url)
		if len(body) > 0 {
			msg = fmt.Sprintf("%s, body: %s", msg, string(body))
		}
		return nil, fmt.Errorf("%s: %s", msg, "unable to create embeddings")
	}

	var response [][]float32
	if err := json.NewDecoder(r.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return response, nil
}
