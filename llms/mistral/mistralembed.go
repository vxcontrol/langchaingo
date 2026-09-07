package mistral

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
)

var (
	ErrEmptyEmbeddings = errors.New("empty embeddings")
	ErrEmbeddingFailed = errors.New("mistral: embedding request failed")
)

const (
	defaultEmbeddingModel = "mistral-embed"
	embeddingsPath        = "v1/embeddings"
)

func convertFloat64ToFloat32(input []float64) []float32 {
	// Create a slice with the same length as the input.
	output := make([]float32, len(input))

	// Iterate over the input slice and convert each element.
	for i, v := range input {
		output[i] = float32(v)
	}

	return output
}

func (m *Model) embeddingsURL() (string, error) {
	endpoint, err := url.Parse(m.clientOptions.endpoint)
	if err != nil {
		return "", fmt.Errorf("%w: %w", ErrEmbeddingFailed, err)
	}
	endpoint.Path = path.Join(endpoint.Path, embeddingsPath)
	return endpoint.String(), nil
}

// CreateEmbedding implements the embeddings.EmbedderClient interface and creates embeddings for the given input texts.
func (m *Model) CreateEmbedding(ctx context.Context, inputTexts []string) ([][]float32, error) {
	model := m.clientOptions.embeddingModel
	if model == "" {
		model = defaultEmbeddingModel
	}

	endpoint, err := m.embeddingsURL()
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(map[string]any{"model": model, "input": inputTexts})
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrEmbeddingFailed, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrEmbeddingFailed, err)
	}
	req.Header.Set("Authorization", "Bearer "+m.clientOptions.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := (&http.Client{Timeout: m.clientOptions.timeout}).Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrEmbeddingFailed, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("%w: %s: %s", ErrEmbeddingFailed, resp.Status, strings.TrimSpace(string(detail)))
	}

	var decoded struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrEmbeddingFailed, err)
	}

	allEmbds := make([][]float32, len(decoded.Data))
	for i, embs := range decoded.Data {
		if len(embs.Embedding) == 0 {
			return nil, ErrEmptyEmbeddings
		}
		allEmbds[i] = convertFloat64ToFloat32(embs.Embedding)
	}
	return allEmbds, nil
}
