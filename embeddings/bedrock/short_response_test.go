package bedrock_test

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/embeddings/bedrock"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/stretchr/testify/require"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func embedderAnswering(t *testing.T, body string) *bedrock.Bedrock {
	t.Helper()

	httpClient := &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(body)),
			Request:    req,
		}, nil
	})}

	cfg, err := config.LoadDefaultConfig(t.Context(),
		config.WithHTTPClient(httpClient),
		config.WithRegion(replayRegion),
		config.WithCredentialsProvider(&fakeCredentialsProvider{}),
	)
	require.NoError(t, err)

	client := bedrockruntime.NewFromConfig(cfg, func(o *bedrockruntime.Options) {
		o.AuthSchemePreference = []string{"sigv4"}
	})

	b, err := bedrock.NewBedrock(bedrock.WithClient(client), bedrock.WithModel(bedrock.ModelCohereEn))
	require.NoError(t, err)

	return b
}

func TestEmbedQueryRefusesAnAnswerWithoutVectors(t *testing.T) {
	t.Parallel()

	b := embedderAnswering(t, `{"response_type":"embeddings_floats","embeddings":[]}`)

	emb, err := b.EmbedQuery(t.Context(), "hi")
	require.ErrorIs(t, err, embeddings.ErrNoEmbedding)
	require.Nil(t, emb)
}

func TestEmbedDocumentsRefusesAShortAnswer(t *testing.T) {
	t.Parallel()

	b := embedderAnswering(t, `{"response_type":"embeddings_floats","embeddings":[[0.1]]}`)

	emb, err := b.EmbedDocuments(t.Context(), []string{"one", "two"})
	require.ErrorIs(t, err, embeddings.ErrShortEmbedding)
	require.Nil(t, emb)
}

func TestEmbedDocumentsKeepsAFullAnswer(t *testing.T) {
	t.Parallel()

	b := embedderAnswering(t, `{"response_type":"embeddings_floats","embeddings":[[0.1],[0.2]]}`)

	emb, err := b.EmbedDocuments(t.Context(), []string{"one", "two"})
	require.NoError(t, err)
	require.Len(t, emb, 2)
}
