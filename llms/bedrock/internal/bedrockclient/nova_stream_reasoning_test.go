package bedrockclient

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTheNovaStreamReadsAReasoningDelta(t *testing.T) {
	t.Parallel()

	var chunk novaStreamingResponseChunk
	require.NoError(t, json.Unmarshal(
		[]byte(`{"contentBlockDelta":{"delta":{"reasoningContent":{"text":"thinking"}}}}`), &chunk))

	require.NotNil(t, chunk.ContentBlockDelta.Delta.ReasoningContent,
		"the vendor sends the thought in this shape; ignoring it drops the thought")
	assert.Equal(t, "thinking", chunk.ContentBlockDelta.Delta.ReasoningContent.Text)
	assert.Empty(t, chunk.ContentBlockDelta.Delta.Text)
}

func TestTheLegacyStreamReadsAnEncryptedBlock(t *testing.T) {
	t.Parallel()

	var chunk streamingCompletionResponseChunk
	require.NoError(t, json.Unmarshal(
		[]byte(`{"type":"content_block_start","content_block":{"type":"redacted_thinking","data":"deadbeef"}}`),
		&chunk))

	assert.Equal(t, "redacted_thinking", chunk.ContentBlock.Type)
	assert.Equal(t, "deadbeef", chunk.ContentBlock.Data,
		"without this field the encrypted thought has nowhere to land")
}
