package dolt_test

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vxcontrol/langchaingo/chains"
	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores"
	"github.com/vxcontrol/langchaingo/vectorstores/dolt"

	"github.com/go-sql-driver/mysql"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

var (
	//nolint:gochecknoglobals
	doltExec string
	//nolint:gochecknoglobals
	doltExecOnce sync.Once
)

type testDoltServer struct {
	t            *testing.T
	Cmd          *exec.Cmd
	db           *sql.DB
	Name         string
	OutputString string
	WaitError    error
	Waited       chan (bool)
	CmdDir       string
	Host         string
	Port         string
	Password     string
}

func newTestDoltServer(t *testing.T) *testDoltServer {
	t.Helper()
	return &testDoltServer{
		t:      t,
		Waited: make(chan bool),
		Name:   "vectorstore_dolt_test",
	}
}

func mustGetDoltExec(t *testing.T) string {
	t.Helper()

	doltCommand := "dolt"
	if runtime.GOOS == "windows" {
		doltCommand = "dolt.exe"
	}

	doltExecOnce.Do(func() {
		arg := os.Getenv("DOLT_BIN")
		if arg == "" {
			arg = doltCommand
		}
		// LookPath checks that DOLT_BIN names an executable, so a wrong one
		// skips the tests too. A name without a separator is searched in PATH,
		// a relative path is taken from the package directory and made
		// absolute, since the dolt commands run in a temporary directory.
		de, err := exec.LookPath(arg)
		if err == nil {
			de, err = filepath.Abs(de)
		}
		if err != nil {
			return
		}
		doltExec = de
	})
	// Skip outside the Once: a skip inside it ends only the first caller, and
	// every later test would run an empty command ("exec: no command").
	if doltExec == "" {
		t.Skip("Dolt binary not available")
	}
	return doltExec
}

// skipIfNoDolt skips a test that starts a server of its own when the dolt
// binary is missing. Tests call it before httprr.OpenForTest, which empties
// the cassette in record mode, so a skipped recording keeps the cassette.
func skipIfNoDolt(t *testing.T) {
	t.Helper()
	if os.Getenv("DOLT_CONNECTION_STRING") == "" {
		mustGetDoltExec(t)
	}
}

// parallelIfOwnServer runs the test in parallel only when it replays and starts
// a server of its own. Tests on the DOLT_CONNECTION_STRING server share its
// tables, and their schema changes fail when they run at once.
func parallelIfOwnServer(t *testing.T, rr *httprr.RecordReplay) {
	t.Helper()
	if !rr.Recording() && os.Getenv("DOLT_CONNECTION_STRING") == "" {
		t.Parallel()
	}
}

func (di *testDoltServer) ConnectionString() string {
	return fmt.Sprintf("%s:%s@(%s:%s)/%s?parseTime=true&multiStatements=true", "root", di.Password, di.Host, di.Port, di.Name)
}

//nolint:funlen
func (di *testDoltServer) Start() error {
	// Look the binary up first: a skip after MkdirTemp leaves the directory behind.
	doltBin := mustGetDoltExec(di.t)

	tmpDir, err := os.MkdirTemp("", "dolt-vectorstore-tests*")
	require.NoError(di.t, err)

	di.CmdDir = tmpDir

	doltInit := exec.Command(doltBin, "init") //nolint:gosec
	doltInit.Env = os.Environ()
	doltInit.Dir = tmpDir
	doltInit.Stdout = os.Stdout
	doltInit.Stderr = os.Stderr
	err = doltInit.Run()
	require.NoError(di.t, err)

	createDB := exec.Command(doltBin, "sql", "-q", fmt.Sprintf("CREATE DATABASE %s;", di.Name)) //nolint:gosec
	createDB.Env = os.Environ()
	createDB.Dir = tmpDir
	createDB.Stdout = os.Stdout
	createDB.Stderr = os.Stderr
	err = createDB.Run()
	require.NoError(di.t, err)

	port, err := getFreePort()
	require.NoError(di.t, err)

	di.Host = "0.0.0.0"
	di.Port = port
	di.Password = ""

	di.Cmd = exec.Command( //nolint:gosec
		doltBin,
		"sql-server",
		"--host", di.Host,
		"--port", di.Port,
	)

	di.Cmd.Env = di.Cmd.Environ()
	di.Cmd.Dir = di.CmdDir

	// exec copies the output into the buffer itself and Wait returns only after
	// that copy is done, so the buffer is complete once Waited is closed. A
	// pipe read by a goroutine of our own races with Wait closing the pipe.
	// Dolt reports a port in use on stdout and logs to stderr.
	var output bytes.Buffer
	di.Cmd.Stdout = &output
	di.Cmd.Stderr = &output

	err = di.Cmd.Start()
	require.NoError(di.t, err)
	go func() {
		di.WaitError = di.Cmd.Wait()
		di.OutputString = output.String()
		close(di.Waited)
	}()

	for i := 0; i < 50; i++ {
		db, err := sql.Open("mysql", di.ConnectionString())
		if err == nil {
			err = db.Ping()
			if err == nil {
				di.db = db
				return nil
			}
			db.Close()
		}
		select {
		case <-di.Waited:
			os.RemoveAll(di.CmdDir)
			return errors.Join(fmt.Errorf("dolt sql-server exited: %s", di.ErrorMessage()), di.WaitError)
		case <-time.After(100 * time.Millisecond):
		}
	}
	shutdownErr := di.Shutdown()
	return errors.Join(fmt.Errorf("dolt sql-server is not accepting connections: %s", di.ErrorMessage()), shutdownErr)
}

func (di *testDoltServer) IsRunning() bool {
	select {
	case <-di.Waited:
		return false
	default:
	}
	return di.db != nil && di.db.Ping() == nil
}

func (di *testDoltServer) Shutdown() error {
	defer os.RemoveAll(di.CmdDir)

	killed := false
	if runtime.GOOS == "windows" {
		kill := exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(di.Cmd.Process.Pid)) //nolint:gosec
		kill.Stdout = os.Stdout
		kill.Stderr = os.Stderr
		err := kill.Run()
		if err != nil {
			return err
		}
		killed = true
	} else {
		err := di.Cmd.Process.Signal(os.Interrupt)
		if err != nil {
			return err
		}
	}
	<-di.Waited
	if killed && di.WaitError != nil {
		return nil
	}
	return di.WaitError
}

func (di *testDoltServer) ErrorMessage() string {
	return di.OutputString
}

func (di *testDoltServer) DB() (*sql.DB, error) {
	if !di.IsRunning() {
		return nil, errors.New("dolt server is not running")
	}
	return di.db, nil
}

func getFreePort() (string, error) {
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		return "", err
	}
	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		return "", err
	}
	defer l.Close()
	addr, ok := l.Addr().(*net.TCPAddr)
	if !ok {
		return "", errors.New("failed to get port")
	}
	return fmt.Sprintf("%d", addr.Port), nil
}

func createOpenAILLM(t *testing.T, rr *httprr.RecordReplay) *openai.LLM {
	t.Helper()

	opts := []openai.Option{
		openai.WithModel("gpt-4.1-nano"),
		openai.WithEmbeddingModel("text-embedding-ada-002"),
		openai.WithHTTPClient(rr.Client()),
	}
	if !rr.Recording() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)

	return llm
}

func createGoogleAIEmbedder(t *testing.T, rr *httprr.RecordReplay) *embeddings.EmbedderImpl {
	t.Helper()

	apiKey := os.Getenv("GOOGLE_API_KEY")
	if !rr.Recording() {
		apiKey = "test-api-key"
	}

	llm, err := googleai.New(
		context.Background(),
		googleai.WithDefaultEmbeddingModel("gemini-embedding-001"),
		googleai.WithHTTPClient(rr.Client()),
		googleai.WithAPIKey(apiKey),
	)
	require.NoError(t, err)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	return e
}

func preCheckEnvSetting(t *testing.T) string {
	t.Helper()

	doltURL := os.Getenv("DOLT_CONNECTION_STRING")
	if doltURL == "" {
		di := newTestDoltServer(t)
		err := di.Start()
		// The free port can be taken before dolt binds it; start on a new one then.
		for attempt := 1; attempt < 3 && err != nil && strings.Contains(err.Error(), "already in use"); attempt++ {
			t.Logf("dolt sql-server port %s is taken, starting on a new one", di.Port)
			di = newTestDoltServer(t)
			err = di.Start()
		}
		if err != nil && strings.Contains(err.Error(), "Cannot connect to the Docker daemon") {
			t.Skip("Docker not available")
		}
		require.NoError(t, err)
		t.Cleanup(func() {
			require.NoError(t, di.Shutdown())
		})
		doltURL = di.ConnectionString()
	}

	return doltURL
}

func makeNewDatabaseName() string {
	return fmt.Sprintf("test-database-%s", uuid.New().String())
}

func cleanupTestArtifacts(ctx context.Context, t *testing.T, s dolt.Store, doltURL string) {
	t.Helper()

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	tx, err := db.BeginTx(ctx, nil)
	require.NoError(t, err)

	require.NoError(t, s.RemoveDatabase(ctx, tx))

	require.NoError(t, tx.Commit())
}

func TestDoltStoreRest(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	llm := createOpenAILLM(t, rr)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(ctx, []schema.Document{
		{PageContent: "tokyo", Metadata: map[string]any{
			"country": "japan",
		}},
		{PageContent: "potato"},
	})
	require.NoError(t, err)

	docs, err := store.SimilaritySearch(ctx, "japan", 1)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.Equal(t, "tokyo", docs[0].PageContent)
	require.Equal(t, "japan", docs[0].Metadata["country"])
}

func TestDoltStoreRestWithScoreThreshold(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	llm := createOpenAILLM(t, rr)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(context.Background(), []schema.Document{
		{PageContent: "Tokyo"},
		{PageContent: "Yokohama"},
		{PageContent: "Osaka"},
		{PageContent: "Nagoya"},
		{PageContent: "Sapporo"},
		{PageContent: "Fukuoka"},
		{PageContent: "Dublin"},
		{PageContent: "Paris"},
		{PageContent: "London"},
		{PageContent: "New York"},
	})
	require.NoError(t, err)

	// test with a score threshold of 0.8, expected 6 documents
	docs, err := store.SimilaritySearch(
		ctx,
		"Which of these are cities in Japan",
		10,
		vectorstores.WithScoreThreshold(0.6), // Dolt uses euclidean squared distance
	)
	require.NoError(t, err)
	require.Len(t, docs, 6)

	// test with a score threshold of 0, expected all 10 documents
	docs, err = store.SimilaritySearch(
		ctx,
		"Which of these are cities in Japan",
		10,
		vectorstores.WithScoreThreshold(0))
	require.NoError(t, err)
	require.Len(t, docs, 10)
}

func TestDoltStoreSimilarityScore(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	llm := createOpenAILLM(t, rr)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(context.Background(), []schema.Document{
		{PageContent: "Tokyo is the capital city of Japan."},
		{PageContent: "Paris is the city of love."},
		{PageContent: "I like to visit London."},
	})
	require.NoError(t, err)

	// Dolt uses euclidean squared distance
	// test with a score threshold of 0.6, expected 6 documents
	docs, err := store.SimilaritySearch(
		ctx,
		"What is the capital city of Japan?",
		3,
		vectorstores.WithScoreThreshold(0.6),
	)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.True(t, docs[0].Score > 0.8)
}

func TestSimilaritySearchWithInvalidScoreThreshold(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	llm := createOpenAILLM(t, rr)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(ctx, []schema.Document{
		{PageContent: "Tokyo"},
		{PageContent: "Yokohama"},
		{PageContent: "Osaka"},
		{PageContent: "Nagoya"},
		{PageContent: "Sapporo"},
		{PageContent: "Fukuoka"},
		{PageContent: "Dublin"},
		{PageContent: "Paris"},
		{PageContent: "London"},
		{PageContent: "New York"},
	})
	require.NoError(t, err)

	_, err = store.SimilaritySearch(
		ctx,
		"Which of these are cities in Japan",
		10,
		vectorstores.WithScoreThreshold(-0.8),
	)
	require.Error(t, err)

	_, err = store.SimilaritySearch(
		ctx,
		"Which of these are cities in Japan",
		10,
		vectorstores.WithScoreThreshold(1.8),
	)
	require.Error(t, err)
}

// Mixing embedders of different sizes in one collection is expected to fail
// on insert. Dolt's vector index, which spans the whole embedding table, is a
// proximity map: a key also sits on an inner level when its hash starts with
// 8 zero bits (1 key in 256; the key holds the random row uuid), and every key
// is filed under the closest key of the level above, by a distance that
// vectors of different lengths do not have. While the index is a single
// non-empty leaf, every edit reaches the root level, so Dolt rebuilds the whole index on each
// flush (ApplyMutationsWithSerializer calls rebuildNode), but
// ProximityMapBuilder.Flush builds a single level without taking a distance,
// and rows of both sizes coexist. What the first inner key changes is that the
// rebuild then files every row under the closest inner key by distance, so the
// 3072-dim Beijing row is compared with a 1536-dim inner key (or the 1536-dim
// rows with Beijing's) and the insert fails; once the index has inner keys, a
// new row is also compared with them on its way down. With the 11 documents
// alone an inner key comes in about 4% of runs. Adding rows of the OpenAI size
// until the index grows an inner level makes it certain: 8192 rows all stay on
// the leaf level with probability (255/256)^8192, about 1e-14. Verified with
// Dolt 1.81.1. The search compares only vectors of the query's size, so it
// still works on the collection.
func TestSimilaritySearchWithDifferentDimensions(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY", "GOOGLE_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	// Avoid issue with different view of request bodies for Google AI SDK
	rr.ScrubReq(httprr.JsonCompactScrubBody)

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	// text-embedding-ada-002 embeds into 1536 dimensions.
	e, err := embeddings.NewEmbedder(createOpenAILLM(t, rr))
	require.NoError(t, err)

	// Zero vectors of the OpenAI size, so the filler rows need no provider.
	filler, err := embeddings.NewEmbedder(embeddings.EmbedderClientFunc(
		func(_ context.Context, texts []string) ([][]float32, error) {
			vectors := make([][]float32, len(texts))
			for i := range vectors {
				vectors[i] = make([]float32, 1536)
			}
			return vectors, nil
		}))
	require.NoError(t, err)

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(ctx, []schema.Document{
		{PageContent: "Tokyo"},
		{PageContent: "Yokohama"},
		{PageContent: "Osaka"},
		{PageContent: "Nagoya"},
		{PageContent: "Sapporo"},
		{PageContent: "Fukuoka"},
		{PageContent: "Dublin"},
		{PageContent: "Paris"},
		{PageContent: "London"},
		{PageContent: "New York"},
	})
	require.NoError(t, err)

	// gemini-embedding-001 embeds into 3072 dimensions.
	_, err = store.AddDocuments(ctx, []schema.Document{
		{PageContent: "Beijing"},
	}, vectorstores.WithEmbedder(createGoogleAIEmbedder(t, rr)))

	fillerDocs := make([]schema.Document, 64)
	for i := range fillerDocs {
		fillerDocs[i] = schema.Document{PageContent: fmt.Sprintf("filler %d", i)}
	}
	for added := 0; err == nil && added < 8192; added += len(fillerDocs) {
		_, err = store.AddDocuments(ctx, fillerDocs, vectorstores.WithEmbedder(filler))
	}

	// Which insert fails depends on the index, and so does the order of the
	// sizes in the message.
	var mysqlErr *mysql.MySQLError
	require.ErrorAs(t, err, &mysqlErr)
	require.Regexp(t,
		`^attempting to find distance between vectors of different lengths: (1536 vs 3072|3072 vs 1536)$`,
		mysqlErr.Message)

	docs, err := store.SimilaritySearch(
		ctx,
		"Which of these are cities in Japan",
		5,
	)
	require.NoError(t, err)
	require.Len(t, docs, 5)
}

func TestDoltAsRetriever(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	llm := createOpenAILLM(t, rr)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(
		ctx,
		[]schema.Document{
			{PageContent: "The color of the house is blue."},
			{PageContent: "The color of the car is red."},
			{PageContent: "The color of the desk is orange."},
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
	require.True(t, strings.Contains(result, "orange"), "expected orange in result")
}

func TestDoltAsRetrieverWithScoreThreshold(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	llm := createOpenAILLM(t, rr)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(
		context.Background(),
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
			vectorstores.ToRetriever(store, 5, vectorstores.WithScoreThreshold(0.7)),
		),
		"What colors are all of the pieces of furniture next to the desk and the desk itself?",
	)
	require.NoError(t, err)

	require.Contains(t, result, "orange", "expected orange in result")
	require.Contains(t, result, "black", "expected black in result")
	require.Contains(t, result, "beige", "expected beige in result")
}

func TestDoltAsRetrieverWithMetadataFilterNotSelected(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	llm := createOpenAILLM(t, rr)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(
		ctx,
		[]schema.Document{
			{
				PageContent: "in kitchen, The color of the lamp beside the desk is black.",
				Metadata: map[string]any{
					"location": "kitchen",
				},
			},
			{
				PageContent: "in bedroom, The color of the lamp beside the desk is blue.",
				Metadata: map[string]any{
					"location": "bedroom",
				},
			},
			{
				PageContent: "in office, The color of the lamp beside the desk is orange.",
				Metadata: map[string]any{
					"location": "office",
				},
			},
			{
				PageContent: "in sitting room, The color of the lamp beside the desk is purple.",
				Metadata: map[string]any{
					"location": "sitting room",
				},
			},
			{
				PageContent: "in patio, The color of the lamp beside the desk is yellow.",
				Metadata: map[string]any{
					"location": "patio",
				},
			},
		},
	)
	require.NoError(t, err)

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store, 5),
		),
		"What color is the lamp in each room?",
	)
	require.NoError(t, err)
	result = strings.ToLower(result)

	require.Contains(t, result, "black", "expected black in result")
	require.Contains(t, result, "blue", "expected blue in result")
	require.Contains(t, result, "orange", "expected orange in result")
	require.Contains(t, result, "purple", "expected purple in result")
	require.Contains(t, result, "yellow", "expected yellow in result")
}

func TestDoltAsRetrieverWithMetadataFilters(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	llm := createOpenAILLM(t, rr)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(
		context.Background(),
		[]schema.Document{
			{
				PageContent: "In office, the color of the lamp beside the desk is orange.",
				Metadata: map[string]any{
					"location":    "office",
					"square_feet": 100,
				},
			},
			{
				PageContent: "in sitting room, the color of the lamp beside the desk is purple.",
				Metadata: map[string]any{
					"location":    "sitting room",
					"square_feet": 400,
				},
			},
			{
				PageContent: "in patio, the color of the lamp beside the desk is yellow.",
				Metadata: map[string]any{
					"location":    "patio",
					"square_feet": 800,
				},
			},
		},
	)
	require.NoError(t, err)

	filter := map[string]any{"location": "sitting room"}

	result, err := chains.Run(
		ctx,
		chains.NewRetrievalQAFromLLM(
			llm,
			vectorstores.ToRetriever(store,
				5,
				vectorstores.WithFilters(filter))),
		"What color is the lamp in each room?",
	)
	require.NoError(t, err)
	require.Contains(t, result, "purple", "expected purple in result")
	require.NotContains(t, result, "orange", "expected not orange in result")
	require.NotContains(t, result, "yellow", "expected not yellow in result")
}

func TestDeduplicater(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	llm := createOpenAILLM(t, rr)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)

	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(context.Background(), []schema.Document{
		{PageContent: "tokyo", Metadata: map[string]any{
			"type": "city",
		}},
		{PageContent: "potato", Metadata: map[string]any{
			"type": "vegetable",
		}},
	}, vectorstores.WithDeduplicater(
		func(_ context.Context, doc schema.Document) bool {
			return doc.PageContent == "tokyo"
		},
	))
	require.NoError(t, err)

	docs, err := store.Search(ctx, 1)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.Equal(t, "potato", docs[0].PageContent)
	require.Equal(t, "vegetable", docs[0].Metadata["type"])
}

func TestWithAllOptions(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "OPENAI_API_KEY")
	skipIfNoDolt(t)

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	parallelIfOwnServer(t, rr)

	doltURL := preCheckEnvSetting(t)
	ctx := context.Background()

	llm := createOpenAILLM(t, rr)
	e, err := embeddings.NewEmbedder(llm)
	require.NoError(t, err)
	require.NoError(t, err)
	db, err := sql.Open("mysql", doltURL)
	require.NoError(t, err)
	defer db.Close()

	store, err := dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
		dolt.WithCollectionTableName("collection_table_name"),
		dolt.WithEmbeddingTableName("embedding_table_name"),
		dolt.WithDatabaseMetadata(map[string]any{
			"key": "value",
		}),
		dolt.WithVectorDimensions(1536),
		dolt.WithCreateEmbeddingIndexAfterAddDocuments(true),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(ctx, []schema.Document{
		{PageContent: "tokyo", Metadata: map[string]any{
			"country": "japan",
		}},
		{PageContent: "potato"},
	})
	require.NoError(t, err)

	docs, err := store.SimilaritySearch(ctx, "japan", 1)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.Equal(t, "tokyo", docs[0].PageContent)
	require.Equal(t, "japan", docs[0].Metadata["country"])

	store, err = dolt.New(
		ctx,
		dolt.WithDB(db),
		dolt.WithEmbedder(e),
		dolt.WithPreDeleteDatabase(true),
		dolt.WithDatabaseName(makeNewDatabaseName()),
		dolt.WithCollectionTableName("collection_table_name1"),
		dolt.WithEmbeddingTableName("embedding_table_name1"),
		dolt.WithDatabaseMetadata(map[string]any{
			"key": "value",
		}),
		dolt.WithVectorDimensions(1536),
		dolt.WithCreateEmbeddingIndexAfterAddDocuments(true),
	)
	require.NoError(t, err)

	defer cleanupTestArtifacts(ctx, t, store, doltURL)

	_, err = store.AddDocuments(ctx, []schema.Document{
		{PageContent: "tokyo", Metadata: map[string]any{
			"country": "japan",
		}},
		{PageContent: "potato"},
	})
	require.NoError(t, err)

	docs, err = store.SimilaritySearch(ctx, "japan", 1)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	require.Equal(t, "tokyo", docs[0].PageContent)
	require.Equal(t, "japan", docs[0].Metadata["country"])
}
