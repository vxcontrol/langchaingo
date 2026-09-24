package openai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCreateEmbeddingRefusesAShortAnswer(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}]}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithToken("t"), WithBaseURL(srv.URL))
	require.NoError(t, err)

	emb, err := llm.CreateEmbedding(t.Context(), []string{"one", "two"})
	require.ErrorIs(t, err, ErrUnexpectedResponseLength)
	require.Nil(t, emb)
}
