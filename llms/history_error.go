package llms

// HistoryFault names what a vendor refused in the history a request replays.
type HistoryFault string

const (
	// HistoryPrefixChanged is a thinking block replayed after something sent
	// before it changed: the system prompt, the tools or an earlier message.
	HistoryPrefixChanged HistoryFault = "prefix_changed"
	// HistorySignatureInvalid is a thinking block whose signature does not
	// verify: truncated, altered or sent back empty.
	HistorySignatureInvalid HistoryFault = "signature_invalid"
	// HistoryThinkingModified is a latest assistant message whose thinking
	// blocks differ from the ones the model returned.
	HistoryThinkingModified HistoryFault = "thinking_modified"
)

// ErrHistoryRejected reports a request the vendor refused for the history it
// replays. Error returns the text of the wrapped error unchanged.
type ErrHistoryRejected struct {
	Fault HistoryFault
	// Message is the index, in the messages passed to GenerateContent, of the
	// message holding the block the vendor named or, when the door merged
	// several messages into the one the vendor named, of the first of them; -1
	// when the door cannot tell: a gateway builds the vendor's messages itself.
	Message int
	Cause   error
}

func (e *ErrHistoryRejected) Error() string { return e.Cause.Error() }

func (e *ErrHistoryRejected) Unwrap() error { return e.Cause }

// ErrContextOverflow reports a request the vendor refused as too large for the
// model's context window or for the API's request size limit. Error returns the
// text of the wrapped error unchanged.
type ErrContextOverflow struct {
	Cause error
}

func (e *ErrContextOverflow) Error() string { return e.Cause.Error() }

func (e *ErrContextOverflow) Unwrap() error { return e.Cause }
