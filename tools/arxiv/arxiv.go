package arxiv

import (
	"context"
	"errors"
	"net/http"

	"github.com/vxcontrol/langchaingo/callbacks"
	"github.com/vxcontrol/langchaingo/tools"
	"github.com/vxcontrol/langchaingo/tools/arxiv/internal"
)

// DefaultUserAgent defines a default value for user-agent header.
const DefaultUserAgent = "github.com/vxcontrol/langchaingo/tools/arxiv"

// Tool defines a tool implementation for the arXiv Search.
type Tool struct {
	CallbacksHandler callbacks.Handler
	client           *internal.Client
}

var _ tools.Tool = Tool{}

type Option func(*Tool)

func WithHTTPClient(client *http.Client) Option {
	return func(t *Tool) {
		t.client.SetHTTPClient(client)
	}
}

// New initializes a new arXiv Search tool with arguments for setting a
// max results per search query and a value for the user agent header.
func New(maxResults int, userAgent string, opts ...Option) (*Tool, error) {
	tool := &Tool{
		client: internal.NewClient(maxResults, userAgent),
	}
	for _, opt := range opts {
		opt(tool)
	}
	return tool, nil
}

// Name returns a name for the tool.
func (t Tool) Name() string {
	return "arXiv Search"
}

// Description returns a description for the tool.
func (t Tool) Description() string {
	return `
	"A wrapper around arXiv Search API."
	"Search for scientific papers on arXiv."
	"Input should be a search query."`
}

// Call performs the search and return the result.
func (t Tool) Call(ctx context.Context, input string) (string, error) {
	if t.CallbacksHandler != nil {
		t.CallbacksHandler.HandleToolStart(ctx, input)
	}

	result, err := t.client.Search(ctx, input)
	if err != nil {
		if errors.Is(err, internal.ErrNoGoodResult) {
			return "No good arXiv Search Results were found", nil
		}
		if t.CallbacksHandler != nil {
			t.CallbacksHandler.HandleToolError(ctx, err)
		}
		return "", err
	}

	if t.CallbacksHandler != nil {
		t.CallbacksHandler.HandleToolEnd(ctx, result)
	}

	return result, nil
}
