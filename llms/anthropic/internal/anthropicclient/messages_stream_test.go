package anthropicclient

import (
	"context"
	"errors"
	"io"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type heldBody struct {
	head   io.Reader
	closed chan struct{}
	once   sync.Once
}

func (b *heldBody) Read(p []byte) (int, error) {
	if n, err := b.head.Read(p); n > 0 || !errors.Is(err, io.EOF) {
		return n, err
	}
	<-b.closed
	return 0, errors.New("read on closed body")
}

func (b *heldBody) Close() error {
	b.once.Do(func() { close(b.closed) })
	return nil
}

func TestAMessagesStreamStopsItsReaderAfterAnErrorEvent(t *testing.T) {
	head := `data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":1}}}` + "\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"sixty"}}` + "\n" +
		`data: {"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}` + "\n"

	runtime.GC()
	time.Sleep(50 * time.Millisecond)
	before := runtime.NumGoroutine()

	body := &heldBody{head: strings.NewReader(head), closed: make(chan struct{})}
	resp, err := parseStreamingMessageResponse(context.Background(), &http.Response{Body: body}, &messagePayload{})
	require.Error(t, err)
	require.NotNil(t, resp)
	require.NoError(t, body.Close())

	deadline := time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > before && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	assert.LessOrEqual(t, runtime.NumGoroutine(), before, "the reader must not stay blocked on a consumer that left")
}
