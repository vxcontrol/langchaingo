package mistral

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func countingServer(t *testing.T, status func(attempt int32) int) (*httptest.Server, *int32) {
	t.Helper()

	var seen int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempt := atomic.AddInt32(&seen, 1)
		code := status(attempt)
		if code != http.StatusOK {
			w.WriteHeader(code)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[{"embedding":[0.1,0.2]}]}`)
	}))
	t.Cleanup(srv.Close)
	return srv, &seen
}

func TestTheEmbeddingCallRetriesAsManyTimesAsTheCallerAsked(t *testing.T) {
	t.Parallel()

	srv, seen := countingServer(t, func(int32) int { return http.StatusInternalServerError })

	m, err := New(WithEndpoint(srv.URL), WithAPIKey("k"), WithMaxRetries(3))
	require.NoError(t, err)
	_, err = m.CreateEmbedding(context.Background(), []string{"x"})

	require.ErrorIs(t, err, ErrEmbeddingFailed)
	assert.Equal(t, int32(3), atomic.LoadInt32(seen),
		"three attempts were asked for, so the vendor sees three requests")
}

func TestARetriedEmbeddingCallReturnsTheVectorItEventuallyGot(t *testing.T) {
	t.Parallel()

	srv, seen := countingServer(t, func(attempt int32) int {
		if attempt < 3 {
			return http.StatusServiceUnavailable
		}
		return http.StatusOK
	})

	m, err := New(WithEndpoint(srv.URL), WithAPIKey("k"), WithMaxRetries(4))
	require.NoError(t, err)
	got, err := m.CreateEmbedding(context.Background(), []string{"x"})

	require.NoError(t, err)
	require.Len(t, got, 1)
	assert.Equal(t, int32(3), atomic.LoadInt32(seen))
}

func TestARefusalTheVendorWillNotChangeIsNotRetried(t *testing.T) {
	t.Parallel()

	srv, seen := countingServer(t, func(int32) int { return http.StatusUnauthorized })

	m, err := New(WithEndpoint(srv.URL), WithAPIKey("k"), WithMaxRetries(5))
	require.NoError(t, err)
	_, err = m.CreateEmbedding(context.Background(), []string{"x"})

	require.ErrorIs(t, err, ErrEmbeddingFailed)
	assert.Equal(t, int32(1), atomic.LoadInt32(seen),
		"401 is the vendor's answer, not a hiccup to sit out")
}

func TestACancelledContextStopsTheRetries(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	var seen int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&seen, 1)
		cancel()
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithEndpoint(srv.URL), WithAPIKey("k"), WithMaxRetries(5))
	require.NoError(t, err)

	started := time.Now()
	_, err = m.CreateEmbedding(ctx, []string{"x"})
	elapsed := time.Since(started)

	require.Error(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(&seen),
		"a cancelled caller does not keep the vendor busy")
	assert.Less(t, elapsed, retryBackoffStep,
		"a cancelled caller gets its answer now, not after the whole backoff ladder")
}

func TestZeroRetriesStillSendsTheRequestOnce(t *testing.T) {
	t.Parallel()

	srv, seen := countingServer(t, func(int32) int { return http.StatusOK })

	m, err := New(WithEndpoint(srv.URL), WithAPIKey("k"), WithMaxRetries(0))
	require.NoError(t, err)
	got, err := m.CreateEmbedding(context.Background(), []string{"x"})

	require.NoError(t, err)
	require.Len(t, got, 1)
	assert.Equal(t, int32(1), atomic.LoadInt32(seen))
}

type recordingTransport struct {
	seen *int32
}

func (r *recordingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	atomic.AddInt32(r.seen, 1)
	return http.DefaultTransport.RoundTrip(req)
}

func TestTheEmbeddingRequestTravelsOnTheCallersClient(t *testing.T) {
	t.Parallel()

	srv, _ := countingServer(t, func(int32) int { return http.StatusOK })

	var through int32
	m, err := New(WithEndpoint(srv.URL), WithAPIKey("k"),
		WithEmbeddingHTTPClient(&http.Client{Transport: &recordingTransport{seen: &through}}))
	require.NoError(t, err)
	_, err = m.CreateEmbedding(context.Background(), []string{"x"})

	require.NoError(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(&through),
		"the caller's transport carries the request, not a client the door made")
}
