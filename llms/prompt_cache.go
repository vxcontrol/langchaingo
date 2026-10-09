package llms

// CacheLayout is how a door places the vendor's prompt cache markers.
type CacheLayout int

const (
	// CacheLayoutDoor leaves the markers to the door's own settings.
	CacheLayoutDoor CacheLayout = iota
	CacheLayoutNone
	// CacheLayoutGrowing places the markers of a history that only grows: on the
	// system prompt, on the start of the current turn and on a block that moves
	// forward with the history for an hour, and on the last block for five minutes.
	CacheLayoutGrowing
)

// WithCacheLayout sets how the door places the vendor's prompt cache markers.
func WithCacheLayout(layout CacheLayout) CallOption {
	return func(o *CallOptions) {
		o.CacheLayout = layout
	}
}
