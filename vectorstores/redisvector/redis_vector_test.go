package redisvector_test

import (
	"context"
	_ "embed"
	"log"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vxcontrol/langchaingo/chains"
	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/ollama"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
	"github.com/vxcontrol/langchaingo/vectorstores/redisvector"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	tcredis "github.com/testcontainers/testcontainers-go/modules/redis"
	"github.com/testcontainers/testcontainers-go/wait"
)

const (
	ollamaModel    = "gemma3:4b"
	embeddingModel = "gemma:2b"
)

func getValues(t *testing.T) (string, string) {
	t.Helper()

	// export OLLAMA_HOST="http://127.0.0.1:11434"
	ollamaURL := os.Getenv("OLLAMA_HOST")
	if ollamaURL == "" {
		t.Skip("OLLAMA_HOST not set")
	}

	uri := os.Getenv("REDIS_URL")
	if uri == "" {
		ctx := t.Context()

		redisContainer, err := tcredis.Run(ctx,
			"redis/redis-stack:7.2.0-v10",
			testcontainers.WithWaitStrategy(
				wait.ForLog("* Ready to accept connections"),
			),
		)
		if err != nil && strings.Contains(err.Error(), "Cannot connect to the Docker daemon") {
			t.Skip("Docker not available")
		}
		require.NoError(t, err)

		t.Cleanup(func() {
			ctx := context.Background() //nolint:usetesting
			require.NoError(t, redisContainer.Terminate(ctx))
		})

		// wait for the container to be ready
		select {
		case <-time.After(5 * time.Second):
		case <-t.Context().Done():
			t.Fatal("test timed out")
		}

		url, err := redisContainer.ConnectionString(ctx)
		if err != nil {
			log.Println(err, redisContainer)
			log.Fatalf("failed to get connection string: %s", err)
		}
		uri = url
	}

	return uri, ollamaURL
}

//go:embed testdata/schema.json
var jsonSchemaData string

//go:embed testdata/schema.yml
var yamlSchemaData string

func TestCreateRedisVectorOptions(t *testing.T) {
	t.Parallel()

	redisURL, ollamaURL := getValues(t)
	_, e := getEmbedding(ollamaURL)
	ctx := t.Context()
	index := "test_case1"

	_, err := redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithEmbedder(e),
	)
	assert.Equal(t, "invalid options: missing index name", err.Error())

	_, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, false),
	)
	assert.Equal(t, "invalid options: missing embedder", err.Error())

	_, err = redisvector.New(ctx,
		redisvector.WithIndexName(index, false),
		redisvector.WithEmbedder(e),
	)
	assert.Equal(t, "redis: invalid URL scheme: ", err.Error())

	_, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, false),
		redisvector.WithEmbedder(e),
	)
	assert.Equal(t, "redis index name does not exist", err.Error())

	_, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
	)
	require.NoError(t, err)

	_, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
		redisvector.WithIndexSchema(redisvector.YAMLSchemaFormat, "./testdata/not_exists.yml", nil),
	)
	assert.Equal(t, "open ./testdata/not_exists.yml: no such file or directory", err.Error())

	_, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
		redisvector.WithIndexSchema(redisvector.YAMLSchemaFormat, "", nil),
	)
	assert.Equal(t, redisvector.ErrEmptySchemaContent, err)

	// create redis vector with file
	_, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
		redisvector.WithIndexSchema(redisvector.YAMLSchemaFormat, "./testdata/schema.yml", nil),
	)
	require.NoError(t, err)

	_, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
		redisvector.WithIndexSchema(redisvector.JSONSchemaFormat, "./testdata/schema.json", nil),
	)
	require.NoError(t, err)

	// create redis vector with string
	_, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
		redisvector.WithIndexSchema(redisvector.JSONSchemaFormat, "", []byte(jsonSchemaData)),
	)
	require.NoError(t, err)

	_, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
		redisvector.WithIndexSchema(redisvector.YAMLSchemaFormat, "", []byte(yamlSchemaData)),
	)
	require.NoError(t, err)
}

func TestAddDocuments(t *testing.T) {
	t.Parallel()

	redisURL, ollamaURL := getValues(t)
	_, e := getEmbedding(ollamaURL)

	ctx := t.Context()

	index := "test_add_document"
	prefix := "doc:"

	_, err := redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, false),
		redisvector.WithEmbedder(e),
	)
	assert.Equal(t, "redis index name does not exist", err.Error())

	vector, err := redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
	)
	require.NoError(t, err)

	err = vector.DropIndex(ctx, index, false)
	assert.Equal(t, "redis index name does not exist", err.Error())

	//nolint: dupl
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
	// create redis vector with not existed index, creating index when adding docs
	docIDs, err := vector.AddDocuments(ctx, data)
	require.NoError(t, err)
	assert.Equal(t, len(data), len(docIDs))
	assert.True(t, strings.HasPrefix(docIDs[0], prefix+index))

	// create data with ids or keys
	dataWithIDOrKeys := []schema.Document{
		{PageContent: "Tokyo", Metadata: map[string]any{"ids": "id1", "population": 9.7, "area": 622}},
		{PageContent: "Kyoto", Metadata: map[string]any{"keys": "key1", "population": 1.46, "area": 828}},
	}

	docIDs, err = vector.AddDocuments(ctx, dataWithIDOrKeys)
	require.NoError(t, err)
	assert.Equal(t, len(dataWithIDOrKeys), len(docIDs))
	assert.Equal(t, prefix+index+":id1", docIDs[0])
	assert.Equal(t, prefix+index+":key1", docIDs[1])

	// create vector with existed index & index schema, will not create new index
	_, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
		redisvector.WithIndexSchema(redisvector.YAMLSchemaFormat, "./testdata/schema.yml", nil),
	)
	require.NoError(t, err)

	// create vector with not existed index & index schema, will create new index with schema
	newIndex := index + "_new"
	vector, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(newIndex, true),
		redisvector.WithEmbedder(e),
		redisvector.WithIndexSchema(redisvector.YAMLSchemaFormat, "./testdata/schema.yml", nil),
	)
	require.NoError(t, err)
	t.Cleanup(func() {
		ctx := context.Background() //nolint:usetesting
		err = vector.DropIndex(ctx, index, true)
		require.NoError(t, err)
		err = vector.DropIndex(ctx, newIndex, true)
		require.NoError(t, err)
	})
}

func TestSimilaritySearch(t *testing.T) {
	t.Parallel()

	redisURL, ollamaURL := getValues(t)
	_, e := getEmbedding(ollamaURL)
	ctx := t.Context()

	index := "test_similarity_search"

	store, err := redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
	)
	require.NoError(t, err)

	//nolint: dupl
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
	// create index and add test data
	docIDs, err := store.AddDocuments(ctx, data)
	require.NoError(t, err)
	assert.Equal(t, len(data), len(docIDs))

	// create vector with existed index
	store, err = redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, false),
		redisvector.WithEmbedder(e),
	)
	require.NoError(t, err)

	docs, err := store.SimilaritySearch(ctx, "Tokyo", 5)
	require.NoError(t, err)
	assert.Len(t, docs, 5)
	assert.Len(t, docs[0].Metadata, 3)

	// search with score threshold
	docs, err = store.SimilaritySearch(ctx, "Tokyo", 10,
		vectorstores.WithScoreThreshold(0.8),
	)
	require.NoError(t, err)
	assert.Len(t, docs, 2)
	assert.Len(t, docs[0].Metadata, 3)

	// search with filter area>1000 or area < 300
	docs, err = store.SimilaritySearch(ctx, "Tokyo", 10,
		vectorstores.WithFilters("(@area:[(1000 +inf] | @area:[-inf (300])"),
	)
	require.NoError(t, err)
	assert.Len(t, docs, 5)
	assert.Len(t, docs[0].Metadata, 3)

	// search with filter area=622
	docs, err = store.SimilaritySearch(ctx, "Tokyo", 10,
		vectorstores.WithFilters("(@area:[622 622])"),
	)
	require.NoError(t, err)
	assert.Len(t, docs, 1)
	assert.Len(t, docs[0].Metadata, 3)

	// search with filter & score threshold
	docs, err = store.SimilaritySearch(ctx, "Tokyo", 2,
		vectorstores.WithFilters("(@area:[(1000 +inf] | @area:[-inf (300])"),
		vectorstores.WithScoreThreshold(0.5),
	)
	require.NoError(t, err)
	assert.Len(t, docs, 2)
	assert.Len(t, docs[0].Metadata, 3)

	t.Cleanup(func() {
		ctx := context.Background() //nolint:usetesting
		err = store.DropIndex(ctx, index, true)
		require.NoError(t, err)
	})
}

func TestRedisVectorAsRetriever(t *testing.T) {
	t.Parallel()

	redisURL, ollamaURL := getValues(t)
	llm, e := getEmbedding(ollamaURL)
	ctx := t.Context()
	index := "test_redis_vector_as_retriever"

	store, err := redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(
		ctx,
		[]schema.Document{
			{PageContent: "The color of the house is blue."},
			{PageContent: "The color of the car is red."},
			{PageContent: "The color of the desk is orange."},
			{PageContent: "The color of the lamp beside the desk is black."},
			{PageContent: "The color of the chair beside the desk is beige."},
		},
	)
	require.NoError(t, err)

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 1),
		),
		"What color is the desk?",
	)
	require.NoError(t, err)
	require.Contains(t, strings.ToLower(result), "orange", "expected orange in result")

	result, err = chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5, vectorstores.WithScoreThreshold(0.2)),
		),
		"What colors are all of the pieces of furniture next to the desk and the desk itself?",
	)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(result), "orange", "expected orange in result")
	require.Contains(t, strings.ToLower(result), "black", "expected black in result")
	require.Contains(t, strings.ToLower(result), "beige", "expected beige in result")

	t.Cleanup(func() {
		ctx := context.Background() //nolint:usetesting
		err = store.DropIndex(ctx, index, true)
		require.NoError(t, err)
	})
}

func TestRedisVectorAsRetrieverWithMetadataFilters(t *testing.T) {
	t.Parallel()

	redisURL, ollamaURL := getValues(t)
	llm, e := getEmbedding(ollamaURL)
	ctx := t.Context()
	index := "test_redis_vector_as_retriever_with_metadata_filters"

	store, err := redisvector.New(ctx,
		redisvector.WithConnectionURL(redisURL),
		redisvector.WithIndexName(index, true),
		redisvector.WithEmbedder(e),
	)
	require.NoError(t, err)

	_, err = store.AddDocuments(
		t.Context(),
		[]schema.Document{
			{
				PageContent: "The color of the lamp beside the desk is black.",
				Metadata: map[string]any{
					"location": "kitchen",
				},
			},
			{
				PageContent: "The color of the lamp beside the desk is blue.",
				Metadata: map[string]any{
					"location": "bedroom",
				},
			},
			{
				PageContent: "The color of the lamp beside the desk is orange.",
				Metadata: map[string]any{
					"location": "office",
				},
			},
			{
				PageContent: "The color of the lamp beside the desk is purple.",
				Metadata: map[string]any{
					"location": "sitting room",
				},
			},
			{
				PageContent: "The color of the lamp beside the desk is yellow.",
				Metadata: map[string]any{
					"location": "patio",
				},
			},
		},
	)
	require.NoError(t, err)
	t.Cleanup(func() {
		ctx := context.Background() //nolint:usetesting
		err = store.DropIndex(ctx, index, true)
		require.NoError(t, err)
	})

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 1,
				vectorstores.WithFilters("@location:(patio)"),
			),
		),
		"What colors is the lamp?",
	)
	require.NoError(t, err)
	require.Contains(t, strings.ToLower(result), "yellow", "expected not yellow in result")
}

// nolint:unparam
func getEmbedding(connectionStr ...string) (llms.Model, *embeddings.EmbedderImpl) {
	opts := []ollama.Option{}
	if len(connectionStr) > 0 {
		opts = append(opts, ollama.WithServerURL(connectionStr[0]))
	}

	llm, err := ollama.New(append(opts, ollama.WithModel(ollamaModel))...)
	if err != nil {
		log.Fatal(err)
	}

	ellm, err := ollama.New(append(opts, ollama.WithModel(embeddingModel))...)
	if err != nil {
		log.Fatal(err)
	}

	e, err := embeddings.NewEmbedder(ellm)
	if err != nil {
		log.Fatal(err)
	}

	return llms.Model(llm), e
}

// nolint:unused
/**
func runOllamaTestContainer(model string) (*tcollama.OllamaContainer, string) {
	ctx := context.Background()

	ollamaContainer, err := tcollama.RunContainer(
		ctx,
		testcontainers.WithImage("ollama/ollama:0.1.31"),
	)
	if err != nil {
		log.Fatalf("failed to start container: %s", err)
	}

	_, _, err = ollamaContainer.Exec(ctx, []string{"ollama", "pull", model})
	if err != nil {
		log.Fatalf("failed to pull model %s: %s", model, err)
	}

	connectionStr, err := ollamaContainer.ConnectionString(ctx)
	if err != nil {
		log.Fatalf("failed to get connection string: %s", err) // nolint:gocritic
	}
	return ollamaContainer, connectionStr
}
*/
