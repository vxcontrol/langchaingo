package llamafile

import (
	"context"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestClient(t *testing.T) *LLM {
	t.Helper()

	options := []Option{
		WithEmbeddingSize(2048),
		WithTemperature(0.8),
	}
	c, err := New(options...)
	require.NoError(t, err)
	return c
}

func TestGenerateContent(t *testing.T) {
	t.Skip("llamafile is not available")
	t.Parallel()

	llm := newTestClient(t)

	parts := []llms.ContentPart{
		llms.TextContent{Text: "Brazil is a country? the answer should just be yes or no"},
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	rsp, err := llm.GenerateContent(t.Context(), content)
	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]
	assert.Regexp(t, "yes", strings.ToLower(c1.Content))
}

func TestWithStreaming(t *testing.T) {
	t.Skip("llamafile is not available")
	t.Parallel()

	llm := newTestClient(t)

	parts := []llms.ContentPart{
		llms.TextContent{Text: "Brazil is a country? answer yes or no"},
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	var (
		sb         strings.Builder
		streamDone bool
	)
	rsp, err := llm.GenerateContent(t.Context(), content,
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			switch chunk.Type { //nolint:exhaustive
			case streaming.ChunkTypeText:
				sb.WriteString(chunk.Content)
			case streaming.ChunkTypeDone:
				streamDone = true
			default:
				// skip other chunks
			}
			return nil
		}))
	require.NoError(t, err)

	assert.True(t, streamDone)
	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]
	assert.Regexp(t, "yes", strings.ToLower(c1.Content))
	assert.Regexp(t, "yes", strings.ToLower(sb.String()))
}

func TestCreateEmbedding(t *testing.T) {
	t.Skip("llamafile is not available")
	t.Parallel()

	llm := newTestClient(t)

	embeddings, err := llm.CreateEmbedding(t.Context(), []string{"hello", "world"})
	require.NoError(t, err)
	assert.Len(t, embeddings, 2)
}
