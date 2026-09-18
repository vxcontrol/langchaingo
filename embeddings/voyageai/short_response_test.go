package voyageai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/embeddings"

	"github.com/stretchr/testify/require"
)

func embedderAnswering(t *testing.T, body string) *VoyageAI {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)

	v, err := NewVoyageAI(WithToken("t"), WithBaseURL(srv.URL))
	require.NoError(t, err)

	return v
}

func TestEmbedQueryRefusesAnAnswerWithoutVectors(t *testing.T) {
	t.Parallel()

	v := embedderAnswering(t, `{"data":[]}`)

	emb, err := v.EmbedQuery(t.Context(), "hi")
	require.ErrorIs(t, err, embeddings.ErrNoEmbedding)
	require.Nil(t, emb)
}

func TestEmbedDocumentsRefusesAShortAnswer(t *testing.T) {
	t.Parallel()

	v := embedderAnswering(t, `{"data":[{"embedding":[0.1]}]}`)

	emb, err := v.EmbedDocuments(t.Context(), []string{"one", "two"})
	require.ErrorIs(t, err, embeddings.ErrShortEmbedding)
	require.Nil(t, emb)
}

func TestEmbedDocumentsKeepsAFullAnswer(t *testing.T) {
	t.Parallel()

	v := embedderAnswering(t, `{"data":[{"embedding":[0.1]},{"embedding":[0.2]}]}`)

	emb, err := v.EmbedDocuments(t.Context(), []string{"one", "two"})
	require.NoError(t, err)
	require.Len(t, emb, 2)
}
