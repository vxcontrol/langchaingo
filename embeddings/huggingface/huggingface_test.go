package huggingface

import (
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms/huggingface"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHuggingfaceEmbeddings(t *testing.T) {
	ctx := t.Context()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
	httprr.SkipIfRecordingMissing(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	// Only run tests in parallel when not recording (to avoid rate limits)
	if rr.Replaying() {
		t.Parallel()
	}

	// Create HuggingFace client with httprr HTTP client
	opts := []huggingface.Option{huggingface.WithHTTPClient(rr.Client())}
	if rr.Replaying() {
		opts = append(opts, huggingface.WithToken("test-api-key"))
	}
	hfClient, err := huggingface.New(opts...)
	require.NoError(t, err)

	e, err := NewHuggingface(WithClient(*hfClient))
	require.NoError(t, err)

	_, err = e.EmbedQuery(ctx, "Hello world!")
	require.NoError(t, err)

	embeddings, err := e.EmbedDocuments(ctx, []string{
		"Hello world",
		"The world is ending",
		"good bye",
	})
	require.NoError(t, err)
	assert.Len(t, embeddings, 3)
}
