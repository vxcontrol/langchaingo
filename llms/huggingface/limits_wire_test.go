package huggingface

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func bodyOfCall(t *testing.T, messages []llms.MessageContent, call ...llms.CallOption) map[string]any {
	t.Helper()

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithToken("t"), WithURL(srv.URL), WithModel("Qwen/Qwen3-32B"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(), messages, call...)
	require.NoError(t, err)

	var body map[string]any
	require.NoError(t, json.Unmarshal(raw, &body))
	return body
}

func TestTheOutputCapOnTheWireComesFromWithMaxTokens(t *testing.T) {
	t.Parallel()

	body := bodyOfCall(t, oneMessage(), llms.WithMaxTokens(500))
	assert.InDelta(t, 500, body["max_tokens"], 0)
}

func TestWithMaxLengthNoLongerReachesTheWire(t *testing.T) {
	t.Parallel()

	body := bodyOfCall(t, oneMessage(), llms.WithMaxLength(20))
	assert.NotContains(t, body, "max_tokens")
}

func TestAnExplicitZeroTemperatureReachesTheWire(t *testing.T) {
	t.Parallel()

	body := bodyOfCall(t, oneMessage(), llms.WithTemperature(0))
	require.Contains(t, body, "temperature")
	assert.InDelta(t, 0, body["temperature"], 0)
}

func TestAnUnsetTemperatureStaysOffTheWire(t *testing.T) {
	t.Parallel()

	body := bodyOfCall(t, oneMessage())
	assert.NotContains(t, body, "temperature")
}

func TestAnExplicitZeroSeedReachesTheWire(t *testing.T) {
	t.Parallel()

	body := bodyOfCall(t, oneMessage(), llms.WithSeed(0))
	require.Contains(t, body, "seed")
	assert.InDelta(t, 0, body["seed"], 0)
}

func TestACallWithoutMessagesIsRefusedInsteadOfPanicking(t *testing.T) {
	t.Parallel()

	llm, err := New(WithToken("t"), WithURL("http://127.0.0.1:1"), WithModel("Qwen/Qwen3-32B"))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(), nil)
	require.ErrorIs(t, err, ErrEmptyMessages)
	require.Nil(t, resp)
}

func TestANonTextPartIsRefusedInsteadOfPanicking(t *testing.T) {
	t.Parallel()

	llm, err := New(WithToken("t"), WithURL("http://127.0.0.1:1"), WithModel("Qwen/Qwen3-32B"))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.ImageURLContent{URL: "https://example.com/a.png"}},
	}})
	require.ErrorIs(t, err, ErrUnsupportedPart)
	require.Nil(t, resp)
}
