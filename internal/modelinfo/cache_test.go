package modelinfo

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAModelIsLookedUpOncePerTTL(t *testing.T) {
	t.Parallel()

	clock := time.Date(2026, 10, 11, 0, 0, 0, 0, time.UTC)
	cache := New[string](time.Hour)
	cache.now = func() time.Time { return clock }
	var calls atomic.Int32
	lookup := func(context.Context) (string, error) {
		calls.Add(1)
		return "reported", nil
	}

	assert.Equal(t, "reported", cache.Get(t.Context(), "claude-opus-6", lookup))
	assert.Equal(t, "reported", cache.Get(t.Context(), "claude-opus-6", lookup))
	assert.Equal(t, int32(1), calls.Load())

	clock = clock.Add(time.Hour)
	assert.Equal(t, "reported", cache.Get(t.Context(), "claude-opus-6", lookup))
	assert.Equal(t, int32(2), calls.Load(), "an answer older than the TTL is asked again")

	assert.Equal(t, "reported", cache.Get(t.Context(), "claude-sonnet-6", lookup))
	assert.Equal(t, int32(3), calls.Load(), "each model is asked for its own answer")
}

func TestAFailedLookupIsKeptAsNothingReported(t *testing.T) {
	t.Parallel()

	cache := New[string](time.Hour)
	var calls atomic.Int32
	failing := func(context.Context) (string, error) {
		calls.Add(1)
		return "partial", errors.New("404 page not found")
	}

	assert.Empty(t, cache.Get(t.Context(), "claude-opus-6", failing))
	assert.Empty(t, cache.Get(t.Context(), "claude-opus-6", failing))
	assert.Equal(t, int32(1), calls.Load(), "a host without the endpoint costs one call per TTL")
}

func TestConcurrentCallsShareOneLookup(t *testing.T) {
	t.Parallel()

	cache := New[string](time.Hour)
	release := make(chan struct{})
	var calls atomic.Int32
	lookup := func(context.Context) (string, error) {
		calls.Add(1)
		<-release
		return "reported", nil
	}

	var wg sync.WaitGroup
	answers := make([]string, 8)
	for i := range answers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			answers[i] = cache.Get(t.Context(), "claude-opus-6", lookup)
		}()
	}
	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()

	assert.Equal(t, int32(1), calls.Load())
	for _, answer := range answers {
		assert.Equal(t, "reported", answer)
	}
}

func TestACallerThatGivesUpDoesNotWaitAndTheAnswerIsKept(t *testing.T) {
	t.Parallel()

	cache := New[string](time.Hour)
	release := make(chan struct{})
	finished := make(chan struct{})
	var calls atomic.Int32
	lookup := func(ctx context.Context) (string, error) {
		calls.Add(1)
		defer close(finished)
		<-release
		return "reported", ctx.Err()
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	assert.Empty(t, cache.Get(ctx, "claude-opus-6", lookup), "a cancelled caller gets nothing at once")

	close(release)
	<-finished
	require.Eventually(t, func() bool {
		return cache.Get(t.Context(), "claude-opus-6", lookup) == "reported"
	}, time.Second, 10*time.Millisecond)
	assert.Equal(t, int32(1), calls.Load(), "the lookup the cancelled caller started answers the next call")
}

func TestANilCacheReportsNothing(t *testing.T) {
	t.Parallel()

	var cache *Cache[string]
	assert.Empty(t, cache.Get(t.Context(), "claude-opus-6", func(context.Context) (string, error) {
		return "reported", nil
	}))
}

func TestAFailedRefreshKeepsTheAnswerItHad(t *testing.T) {
	t.Parallel()

	clock := time.Date(2026, 10, 11, 0, 0, 0, 0, time.UTC)
	cache := New[string](time.Hour)
	cache.now = func() time.Time { return clock }
	failing := false
	lookup := func(context.Context) (string, error) {
		if failing {
			return "", errors.New("503 service unavailable")
		}
		return "reported", nil
	}

	assert.Equal(t, "reported", cache.Get(t.Context(), "claude-opus-6", lookup))
	clock, failing = clock.Add(time.Hour), true
	assert.Equal(t, "reported", cache.Get(t.Context(), "claude-opus-6", lookup))
	clock = clock.Add(30 * time.Minute)
	assert.Equal(t, "reported", cache.Get(t.Context(), "claude-opus-6", lookup))
}

func TestASlowLookupLeavesTheCallerHalfItsDeadline(t *testing.T) {
	t.Parallel()

	cache := New[string](time.Hour)
	release := make(chan struct{})
	lookup := func(context.Context) (string, error) {
		<-release
		return "reported", nil
	}

	ctx, cancel := context.WithTimeout(t.Context(), 400*time.Millisecond)
	defer cancel()
	start := time.Now()
	assert.Empty(t, cache.Get(ctx, "claude-opus-6", lookup))
	assert.Less(t, time.Since(start), 300*time.Millisecond)
	require.NoError(t, ctx.Err(), "the caller keeps time for its own request")

	close(release)
	require.Eventually(t, func() bool {
		return cache.Get(t.Context(), "claude-opus-6", lookup) == "reported"
	}, time.Second, 10*time.Millisecond, "the lookup still answers the next call")
}
