package openaiclient

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAPartThatIsNotTextKeepsItsPlaceAfterTheThinkingChunk(t *testing.T) {
	t.Parallel()

	raw, err := json.Marshal(ChatMessage{Role: "assistant", Thinking: "seen", MultiContent: []llms.ContentPart{
		llms.TextContent{Text: "look"},
		llms.ImageURLContent{URL: "https://example.com/a.png"},
	}})
	require.NoError(t, err)

	assert.JSONEq(t, `{"role":"assistant","content":[`+
		`{"type":"thinking","thinking":[{"type":"text","text":"seen"}]},`+
		`{"type":"text","text":"look"},`+
		`{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}`, string(raw))
}
