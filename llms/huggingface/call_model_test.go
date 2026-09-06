package huggingface

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestTheModelOfTheCallReachesTheRequest(t *testing.T) {
	t.Parallel()

	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"generated_text":"hi"}]`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithToken("t"), WithURL(srv.URL), WithModel("gpt2"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithModel("meta-llama/Llama-3.1-8B-Instruct"))
	require.NoError(t, err)

	assert.Contains(t, path, "meta-llama/Llama-3.1-8B-Instruct",
		"the model named by the call must address the request, not the one from the constructor")
}

func TestWithoutAModelOnTheCallTheConstructorsModelIsUsed(t *testing.T) {
	t.Parallel()

	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"generated_text":"hi"}]`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithToken("t"), WithURL(srv.URL), WithModel("gpt2"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.NoError(t, err)

	assert.Contains(t, path, "gpt2")
}
