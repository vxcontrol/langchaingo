package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestTheTopKPassedToTheLLMReachesTheConverseRequest(t *testing.T) {
	t.Parallel()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, converseAnswer)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv,
		bedrock.WithModel("anthropic.claude-haiku-4-5-20251001-v1:0"), bedrock.WithConverseAPI())
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithTopK(40))
	require.NoError(t, err)

	var sent struct {
		Fields map[string]any `json:"additionalModelRequestFields"`
	}
	require.NoError(t, json.Unmarshal(body, &sent))
	assert.Equal(t, map[string]any{"top_k": float64(40)}, sent.Fields)
	assert.Empty(t, resp.Warnings)
}
