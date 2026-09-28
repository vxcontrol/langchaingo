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
	"time"
)

var (
	ErrEmptyEmbeddings = errors.New("empty embeddings")
	ErrShortEmbeddings = errors.New("mistral: fewer embeddings than inputs")
	ErrEmbeddingFailed = errors.New("mistral: embedding request failed")
)

var retryEmbeddingStatus = map[int]bool{
	http.StatusTooManyRequests:     true,
	http.StatusInternalServerError: true,
	http.StatusBadGateway:          true,
	http.StatusServiceUnavailable:  true,
	http.StatusGatewayTimeout:      true,
}

const (
	retryBackoffStep      = 500 * time.Millisecond
	defaultEmbeddingModel = "mistral-embed"
	apiVersionSegment     = "v1"
	embeddingsSegment     = "embeddings"
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
	if path.Base(endpoint.Path) != apiVersionSegment {
		endpoint.Path = path.Join(endpoint.Path, apiVersionSegment)
	}
	endpoint.Path = path.Join(endpoint.Path, embeddingsSegment)
	return endpoint.String(), nil
}

func (m *Model) embeddingsHTTPClient() *http.Client {
	if m.clientOptions.embeddingHTTPClient != nil {
		return m.clientOptions.embeddingHTTPClient
	}
	return &http.Client{Timeout: m.clientOptions.timeout}
}

func (m *Model) postEmbeddings(ctx context.Context, endpoint string, body []byte) (*http.Response, error) {
	client := m.embeddingsHTTPClient()

	attempts := m.clientOptions.maxRetries
	if attempts < 1 {
		attempts = 1
	}

	var lastErr error
	for attempt := range attempts {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("%w: %w", ErrEmbeddingFailed, ctx.Err())
			case <-time.After(time.Duration(attempt) * retryBackoffStep):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("%w: %w", ErrEmbeddingFailed, err)
		}
		req.Header.Set("Authorization", "Bearer "+m.clientOptions.apiKey)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("%w: %w", ErrEmbeddingFailed, err)
			continue
		}
		if !retryEmbeddingStatus[resp.StatusCode] {
			return resp, nil
		}

		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		lastErr = fmt.Errorf("%w: %s: %s", ErrEmbeddingFailed, resp.Status, strings.TrimSpace(string(detail)))
	}

	return nil, lastErr
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

	resp, err := m.postEmbeddings(ctx, endpoint, body)
	if err != nil {
		return nil, err
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

	if len(decoded.Data) == 0 {
		return nil, ErrEmptyEmbeddings
	}
	if len(decoded.Data) != len(inputTexts) {
		return nil, ErrShortEmbeddings
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
