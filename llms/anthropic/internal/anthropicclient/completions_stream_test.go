package anthropicclient

import (
	"context"
	"errors"
	"io"
	"net/http"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func legacyStream(t *testing.T, body string) (*CompletionResponsePayload, error) {
	t.Helper()
	resp := &http.Response{Body: io.NopCloser(strings.NewReader(body))}
	return parseStreamingCompletionResponse(context.Background(), resp, &completionPayload{})
}

func TestALegacyLineLongerThanTheDefaultBufferSurvives(t *testing.T) {
	t.Parallel()

	answer := strings.Repeat("x", 200*1024)
	got, err := legacyStream(t, `data: {"completion":"`+answer+`","model":"claude-2"}`+"\n")

	require.NoError(t, err)
	require.NotNil(t, got)
	assert.Equal(t, answer, got.Completion, "a line over the 64KiB default must not be truncated")
}

func TestALegacyStreamWithNoEventsIsAnError(t *testing.T) {
	t.Parallel()

	got, err := legacyStream(t, ": keep-alive\n\n")

	require.ErrorIs(t, err, ErrEmptyResponse)
	assert.Nil(t, got)
}

func TestALegacyLineOverTheCeilingReportsTheFailure(t *testing.T) {
	t.Parallel()

	got, err := legacyStream(t, `data: {"completion":"`+strings.Repeat("x", maxStreamLine+1)+`"}`+"\n")

	require.ErrorContains(t, err, "failed to read stream",
		"the read failure must reach the caller by name, not as an empty-response error")
	require.NotErrorIs(t, err, ErrEmptyResponse)
	assert.Nil(t, got)
}

func TestALegacyStreamStopsItsReaderWhenTheConsumerGivesUp(t *testing.T) {
	body := strings.Repeat(`data: {"completion":"x","model":"claude-2"}`+"\n", 5)
	gaveUp := errors.New("consumer gave up")

	runtime.GC()
	time.Sleep(50 * time.Millisecond)
	before := runtime.NumGoroutine()

	resp := &http.Response{Body: io.NopCloser(strings.NewReader(body))}
	_, err := parseStreamingCompletionResponse(context.Background(), resp, &completionPayload{
		StreamingFunc: func(context.Context, streaming.Chunk) error { return gaveUp },
	})
	require.ErrorIs(t, err, gaveUp)

	deadline := time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > before && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	assert.LessOrEqual(t, runtime.NumGoroutine(), before, "the reader must not stay blocked on a consumer that left")
}
