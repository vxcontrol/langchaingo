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

func modelOnTheWire(t *testing.T, call ...llms.CallOption) string {
	t.Helper()

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithToken("t"), WithURL(srv.URL), WithModel("gpt2"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, call...)
	require.NoError(t, err)

	var body struct {
		Model string `json:"model"`
	}
	require.NoError(t, json.Unmarshal(raw, &body))

	return body.Model
}

func TestTheModelOfTheCallReachesTheRequest(t *testing.T) {
	t.Parallel()

	model := modelOnTheWire(t, llms.WithModel("meta-llama/Llama-3.1-8B-Instruct"))

	assert.Equal(t, "meta-llama/Llama-3.1-8B-Instruct", model,
		"the model named by the call must address the request, not the one from the constructor")
}

func TestWithoutAModelOnTheCallTheConstructorsModelIsUsed(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "gpt2", modelOnTheWire(t))
}
