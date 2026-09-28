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

func converseInferenceConfig(t *testing.T, opts ...llms.CallOption) map[string]any {
	t.Helper()

	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},`+
			`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv,
		bedrock.WithModel("amazon.nova-lite-v1:0"), bedrock.WithConverseAPI())

	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")}, opts...)
	require.NoError(t, err)

	var payload struct {
		InferenceConfig map[string]any `json:"inferenceConfig"`
	}
	require.NoError(t, json.Unmarshal([]byte(body), &payload))
	return payload.InferenceConfig
}

func TestConverseLeavesOutATokenLimitTheVendorWouldReject(t *testing.T) {
	t.Parallel()

	t.Run("a limit of zero is left out", func(t *testing.T) {
		t.Parallel()

		config := converseInferenceConfig(t, llms.WithMaxTokens(0))

		_, present := config["maxTokens"]
		assert.False(t, present,
			"the vendor accepts a minimum of 1 and defaults to the model's maximum when the field is absent")
	})

	t.Run("a limit the caller set travels", func(t *testing.T) {
		t.Parallel()

		config := converseInferenceConfig(t, llms.WithMaxTokens(64))

		assert.Equal(t, float64(64), config["maxTokens"])
	})
}
