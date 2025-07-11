package cloudsql_test

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/internal/testutil/testctr"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/util/cloudsqlutil"
	"github.com/vxcontrol/langchaingo/vectorstores/cloudsql"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/log"
	tcpostgres "github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
)

func preCheckEnvSetting(t *testing.T) string {
	t.Helper()
	testctr.SkipIfDockerNotAvailable(t)

	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	ctx := t.Context()

	pgvectorURL := os.Getenv("PGVECTOR_CONNECTION_STRING")
	if pgvectorURL == "" {
		pgVectorContainer, err := tcpostgres.Run(
			ctx,
			"docker.io/pgvector/pgvector:pg16",
			tcpostgres.WithDatabase("db_test"),
			tcpostgres.WithUsername("user"),
			tcpostgres.WithPassword("passw0rd!"),
			testcontainers.WithLogger(log.TestLogger(t)),
			testcontainers.WithWaitStrategy(
				wait.ForAll(
					wait.ForLog("database system is ready to accept connections").
						WithOccurrence(2).
						WithStartupTimeout(60*time.Second),
					wait.ForListeningPort("5432/tcp").
						WithStartupTimeout(60*time.Second),
				)),
		)
		if err != nil && strings.Contains(err.Error(), "Cannot connect to the Docker daemon") {
			t.Skip("Docker not available")
		}
		require.NoError(t, err)

		t.Cleanup(func() {
			if err := pgVectorContainer.Terminate(context.Background()); err != nil { //nolint:usetesting
				t.Logf("Failed to terminate cloudsql container: %v", err)
			}
		})

		str, err := pgVectorContainer.ConnectionString(ctx, "sslmode=disable")
		require.NoError(t, err)

		pgvectorURL = str

		// Give the container a moment to fully initialize
		time.Sleep(2 * time.Second)
	}

	return pgvectorURL
}

func setEngineWithImage(t *testing.T) cloudsqlutil.PostgresEngine {
	t.Helper()

	pgvectorURL := preCheckEnvSetting(t)
	ctx := t.Context()
	myPool, err := pgxpool.New(ctx, pgvectorURL)
	if err != nil {
		t.Fatal("Could not set Engine: ", err)
	}
	// Call NewPostgresEngine to initialize the database connection
	pgEngine, err := cloudsqlutil.NewPostgresEngine(ctx,
		cloudsqlutil.WithPool(myPool),
	)
	if err != nil {
		t.Fatal("Could not set Engine: ", err)
	}

	return pgEngine
}

// createOpenAIEmbedder creates an OpenAI embedder with httprr support for testing.
func createOpenAIEmbedder(t *testing.T) *embeddings.EmbedderImpl {
	t.Helper()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	opts := []openai.Option{
		openai.WithEmbeddingModel("text-embedding-ada-002"),
		openai.WithHTTPClient(rr.Client()),
	}
	if !rr.Recording() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}

	llm, err := openai.New(opts...)
	if err != nil {
		t.Fatalf("Failed to create OpenAI LLM: %v", err)
	}
	e, err := embeddings.NewEmbedder(llm)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}
	return e
}

func initVectorStore(t *testing.T) (cloudsql.VectorStore, func() error) {
	t.Helper()

	pgEngine := setEngineWithImage(t)
	ctx := t.Context()
	vectorstoreTableoptions := cloudsqlutil.VectorstoreTableOptions{
		TableName:         "my_test_table",
		OverwriteExisting: true,
		VectorSize:        1536,
		StoreMetadata:     true,
	}
	err := pgEngine.InitVectorstoreTable(ctx, vectorstoreTableoptions)
	if err != nil {
		t.Fatal(err)
	}

	// Initialize OpenAI embedder with httprr
	e := createOpenAIEmbedder(t)
	vs, err := cloudsql.NewVectorStore(pgEngine, e, "my_test_table")
	if err != nil {
		t.Fatal(err)
	}

	cleanUpTableFn := func() error {
		_, err := pgEngine.Pool.Exec(ctx, fmt.Sprintf("DROP TABLE IF EXISTS %s", "my_test_table"))
		return err
	}
	return vs, cleanUpTableFn
}

func TestContainerPingToDB(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	engine := setEngineWithImage(t)

	defer engine.Close()

	if err := engine.Pool.Ping(ctx); err != nil {
		t.Fatal(err)
	}
}

func TestContainerApplyVectorIndexAndDropIndex(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	vs, cleanUpTableFn := initVectorStore(t)
	idx := vs.NewBaseIndex(
		"testindex",
		"hnsw",
		cloudsql.CosineDistance{},
		[]string{},
		cloudsql.HNSWOptions{M: 4, EfConstruction: 16},
	)
	err := vs.ApplyVectorIndex(ctx, idx, "testindex", false)
	if err != nil {
		t.Fatal(err)
	}
	err = vs.DropVectorIndex(ctx, "testindex")
	if err != nil {
		t.Fatal(err)
	}
	err = cleanUpTableFn()
	if err != nil {
		t.Fatal(err)
	}
}

func TestContainerIsValidIndex(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	vs, cleanUpTableFn := initVectorStore(t)
	idx := vs.NewBaseIndex(
		"testindex",
		"hnsw",
		cloudsql.CosineDistance{},
		[]string{},
		cloudsql.HNSWOptions{M: 4, EfConstruction: 16},
	)
	err := vs.ApplyVectorIndex(ctx, idx, "testindex", false)
	if err != nil {
		t.Fatal(err)
	}
	_, err = vs.IsValidIndex(ctx, "testindex")
	if err != nil {
		t.Fatal(err)
	}
	err = vs.DropVectorIndex(ctx, "testindex")
	if err != nil {
		t.Fatal(err)
	}
	err = cleanUpTableFn()
	if err != nil {
		t.Fatal(err)
	}
}

func TestContainerAddDocuments(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()
	if !rr.Recording() {
		t.Parallel()
	}

	ctx := t.Context()
	vs, cleanUpTableFn := initVectorStore(t)

	_, err := vs.AddDocuments(ctx, []schema.Document{
		{
			PageContent: "Tokyo",
			Metadata: map[string]any{
				"population": 38,
				"area":       2190,
			},
		},
		{
			PageContent: "Paris",
			Metadata: map[string]any{
				"population": 11,
				"area":       105,
			},
		},
		{
			PageContent: "London",
			Metadata: map[string]any{
				"population": 9.5,
				"area":       1572,
			},
		},
		{
			PageContent: "Santiago",
			Metadata: map[string]any{
				"population": 6.9,
				"area":       641,
			},
		},
		{
			PageContent: "Buenos Aires",
			Metadata: map[string]any{
				"population": 15.5,
				"area":       203,
			},
		},
		{
			PageContent: "Rio de Janeiro",
			Metadata: map[string]any{
				"population": 13.7,
				"area":       1200,
			},
		},
		{
			PageContent: "Sao Paulo",
			Metadata: map[string]any{
				"population": 22.6,
				"area":       1523,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	err = cleanUpTableFn()
	if err != nil {
		t.Fatal(err)
	}
}
