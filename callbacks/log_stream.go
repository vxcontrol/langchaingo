//nolint:forbidigo
package callbacks

import (
	"context"
	"fmt"

	"github.com/vxcontrol/langchaingo/llms/streaming"
)

// StreamLogHandler is a callback handler that prints to the standard output streaming.
type StreamLogHandler struct {
	SimpleHandler
}

var _ Handler = StreamLogHandler{}

func (StreamLogHandler) HandleStreamingFunc(_ context.Context, chunk streaming.Chunk) {
	fmt.Println(chunk.String())
}
