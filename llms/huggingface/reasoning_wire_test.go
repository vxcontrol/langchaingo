package huggingface

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
)

func effortOnTheWire(t *testing.T, call ...llms.CallOption) (string, bool) {
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

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, call...)
	require.NoError(t, err)

	var body map[string]any
	require.NoError(t, json.Unmarshal(raw, &body))
	effort, present := body["reasoning_effort"]
	if !present {
		return "", false
	}

	return effort.(string), true
}

func TestTheAskedForEffortReachesTheWireUnchanged(t *testing.T) {
	t.Parallel()

	effort, present := effortOnTheWire(t, llms.WithReasoning(llms.ReasoningHigh, 0))

	require.True(t, present)
	assert.Equal(t, "high", effort)
}

func TestSwitchingThinkingOffSendsTheDocumentedNone(t *testing.T) {
	t.Parallel()

	effort, present := effortOnTheWire(t, llms.WithReasoningDisabled())

	require.True(t, present)
	assert.Equal(t, "none", effort)
}

func TestACallThatSaidNothingAboutThinkingCarriesNoEffort(t *testing.T) {
	t.Parallel()

	_, present := effortOnTheWire(t)

	assert.False(t, present, "a caller who said nothing must not have an effort invented for them")
}
