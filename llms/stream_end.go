package llms

import "errors"

var (
	// ErrIncompleteStream reports a response stream that ended before its final
	// event; the response returned with it holds what arrived before the end.
	ErrIncompleteStream = errors.New("llms: response stream ended before its final event")

	// ErrStreamFailed reports an error the provider sent inside a response stream;
	// the response returned with it holds what arrived before the error.
	ErrStreamFailed = errors.New("llms: provider reported an error inside the response stream")
)
