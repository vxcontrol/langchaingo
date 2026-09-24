package llms

import (
	"bytes"
	"errors"
	"log"
	"sync/atomic"
	"testing"

	"github.com/pkoukk/tiktoken-go"
	"github.com/stretchr/testify/assert"
)

func TestCountTokens(t *testing.T) {
	t.Parallel()
	numTokens := CountTokens("gpt-3.5-turbo", "test for counting tokens")
	expectedNumTokens := 4
	assert.Equal(t, expectedNumTokens, numTokens)
}

// failingBpeLoader stands in for the download of an encoding.
type failingBpeLoader struct {
	calls atomic.Int32
}

func (l *failingBpeLoader) LoadTiktokenBpe(string) (map[string]int, error) {
	l.calls.Add(1)
	return nil, errors.New("encoding download disabled")
}

// TestCountTokensApproximation swaps the package-wide tiktoken loader and log
// output, so it does not run in parallel.
func TestCountTokensApproximation(t *testing.T) {
	loader := &failingBpeLoader{}
	tiktoken.SetBpeLoader(loader)
	var logs bytes.Buffer
	logWriter := log.Writer()
	log.SetOutput(&logs)
	t.Cleanup(func() {
		tiktoken.SetBpeLoader(tiktoken.NewDefaultBpeLoader())
		log.SetOutput(logWriter)
	})

	text := "naïve café" // 10 runes in 12 bytes

	// A model tiktoken does not map is approximated without a download or a warning.
	assert.Equal(t, 2, CountTokens("", text))
	assert.Equal(t, 2, CountTokens("claude-sonnet-4", text))
	assert.Zero(t, loader.calls.Load())
	assert.Empty(t, logs.String())

	// A mapped model whose encoding cannot be loaded is approximated with a warning.
	assert.Equal(t, 2, CountTokens("text-davinci-001", text))
	assert.Equal(t, int32(1), loader.calls.Load())
	assert.Contains(t, logs.String(), "[WARN] Failed to load the token encoding for model text-davinci-001")
}

func TestGetModelContextSize(t *testing.T) {
	t.Parallel()
	tests := []struct {
		model        string
		expectedSize int
	}{
		// GPT-3.5 models
		{"gpt-3.5-turbo", 16385},
		{"gpt-3.5-turbo-16k", 16385},
		{"gpt-3.5-turbo-0125", 16385},
		{"gpt-3.5-turbo-1106", 16385},
		// GPT-4 models
		{"gpt-4", 8192},
		{"gpt-4-32k", 32768},
		{"gpt-4-0613", 8192},
		{"gpt-4-32k-0613", 32768},
		// GPT-4 Turbo models
		{"gpt-4-turbo", 128000},
		{"gpt-4-turbo-preview", 128000},
		{"gpt-4-turbo-2024-04-09", 128000},
		{"gpt-4-1106-preview", 128000},
		{"gpt-4-0125-preview", 128000},
		// GPT-4o models
		{"gpt-4o", 128000},
		{"gpt-4o-2024-05-13", 128000},
		{"gpt-4o-2024-08-06", 128000},
		{"gpt-4o-mini", 128000},
		{"gpt-4o-mini-2024-07-18", 128000},
		// Legacy models
		{"text-davinci-003", 4097},
		{"text-curie-001", 2048},
		{"text-babbage-001", 2048},
		{"text-ada-001", 2048},
		{"code-davinci-002", 8000},
		{"code-cushman-001", 2048},
		// Unknown model should return default
		{"unknown-model", 2048},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			size := GetModelContextSize(tt.model)
			assert.Equal(t, tt.expectedSize, size, "Context size for model %s", tt.model)
		})
	}
}
