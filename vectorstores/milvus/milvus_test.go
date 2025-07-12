package milvus

import (
	"context"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/internal/testutil/testctr"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	tclog "github.com/testcontainers/testcontainers-go/log"
	tcmilvus "github.com/testcontainers/testcontainers-go/modules/milvus"
)

// createOpenAIEmbedder creates an OpenAI embedder with httprr support for testing.
func createOpenAIEmbedder(t *testing.T) (llms.Model, *embeddings.EmbedderImpl) {
	t.Helper()
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	t.Cleanup(func() { rr.Close() })

	apiKey := "test-api-key"
	if key := os.Getenv("OPENAI_API_KEY"); key != "" && rr.Recording() {
		apiKey = key
	}

	llm, err := openai.New(
		openai.WithToken(apiKey),
		openai.WithEmbeddingModel("text-embedding-ada-002"),
		openai.WithHTTPClient(rr.Client()),
	)
	require.NoError(t, err)

	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)
	return llms.Model(llm), e
}

func getNewStore(t *testing.T, opts ...Option) (Store, error) {
	t.Helper()
	testctr.SkipIfDockerNotAvailable(t)

	_, e := createOpenAIEmbedder(t)
	ctx := t.Context()
	url := os.Getenv("MILVUS_URL")
	if url == "" {
		image := "milvusdb/milvus:v2.5.14"
		milvusContainer, err := tcmilvus.Run(ctx, image, testcontainers.WithLogger(tclog.TestLogger(t)))
		if err != nil && strings.Contains(err.Error(), "Cannot connect to the Docker daemon") {
			t.Skip("Docker not available")
		}
		require.NoError(t, err)
		t.Cleanup(func() {
			ctx := context.Background() //nolint:usetesting
			if err := milvusContainer.Terminate(ctx); err != nil {
				t.Logf("Failed to terminate milvus container: %v", err)
			}
		})

		url, err = milvusContainer.ConnectionString(ctx)
		if err != nil {
			t.Skipf("Failed to get milvus container endpoint: %s", err)
		}
	}
	config := client.Config{
		Address: url,
	}
	idx, err := entity.NewIndexAUTOINDEX(entity.L2)
	if err != nil {
		return Store{}, err
	}
	opts = append(
		opts,
		WithEmbedder(e),
		WithIndex(idx))
	return New(
		ctx,
		config,
		opts...,
	)
}

func TestMilvusConnection(t *testing.T) {
	ctx := t.Context()
	t.Parallel()
	if testing.Short() {
		t.Skip("Skipping Milvus connection test in short mode")
	}
	storer, err := getNewStore(t, WithDropOld(), WithCollectionName("test"))
	require.NoError(t, err)

	data := []schema.Document{
		{PageContent: "Tokyo", Metadata: map[string]any{"population": 9.7, "area": 622}},
		{PageContent: "Kyoto", Metadata: map[string]any{"population": 1.46, "area": 828}},
		{PageContent: "Hiroshima", Metadata: map[string]any{"population": 1.2, "area": 905}},
		{PageContent: "Kazuno", Metadata: map[string]any{"population": 0.04, "area": 707}},
		{PageContent: "Nagoya", Metadata: map[string]any{"population": 2.3, "area": 326}},
		{PageContent: "Toyota", Metadata: map[string]any{"population": 0.42, "area": 918}},
		{PageContent: "Fukuoka", Metadata: map[string]any{"population": 1.59, "area": 341}},
		{PageContent: "Paris", Metadata: map[string]any{"population": 11, "area": 105}},
		{PageContent: "London", Metadata: map[string]any{"population": 9.5, "area": 1572}},
		{PageContent: "Santiago", Metadata: map[string]any{"population": 6.9, "area": 641}},
		{PageContent: "Buenos Aires", Metadata: map[string]any{"population": 15.5, "area": 203}},
		{PageContent: "Rio de Janeiro", Metadata: map[string]any{"population": 13.7, "area": 1200}},
		{PageContent: "Sao Paulo", Metadata: map[string]any{"population": 22.6, "area": 1523}},
	}

	_, err = storer.AddDocuments(ctx, data)
	require.NoError(t, err)

	// search docs with filter
	filterRes, err := storer.SimilaritySearch(ctx,
		"Tokyo", 10,
		vectorstores.WithFilters("meta['area']==622"),
	)

	require.NoError(t, err)
	require.Len(t, filterRes, 1)

	japanRes, err := storer.SimilaritySearch(ctx,
		"Tokyo", 2,
		vectorstores.WithScoreThreshold(0.5))
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(japanRes), 1)
	require.LessOrEqual(t, len(japanRes), 2)
}
