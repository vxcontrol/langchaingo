package bedrock_test

import (
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/embeddings/bedrock"
	"github.com/vxcontrol/langchaingo/httputil"
	"github.com/vxcontrol/langchaingo/internal/httprr"

	"github.com/stretchr/testify/require"
)

func TestEmbedQuery(t *testing.T) {
	ctx := t.Context()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "AWS_ACCESS_KEY_ID")

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	// Replace httputil.DefaultClient with httprr client
	oldClient := httputil.DefaultClient
	httputil.DefaultClient = rr.Client()
	defer func() { httputil.DefaultClient = oldClient }()

	model, err := bedrock.NewBedrock(bedrock.WithModel(bedrock.ModelTitanEmbedG1))
	require.NoError(t, err)
	_, err = model.EmbedQuery(ctx, "hello world")

	require.NoError(t, err)
}

func TestEmbedDocuments(t *testing.T) {
	ctx := t.Context()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "AWS_ACCESS_KEY_ID")

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	// Replace httputil.DefaultClient with httprr client
	oldClient := httputil.DefaultClient
	httputil.DefaultClient = rr.Client()
	defer func() { httputil.DefaultClient = oldClient }()

	model, err := bedrock.NewBedrock(bedrock.WithModel(bedrock.ModelCohereEn))
	require.NoError(t, err)

	embeddings, err := model.EmbedDocuments(ctx, []string{"hello world", "goodbye world"})

	require.NoError(t, err)
	require.Len(t, embeddings, 2)
}
