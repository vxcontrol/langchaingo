package llms

import "strconv"

// CacheLayout is how a door places the vendor's prompt cache markers.
type CacheLayout int

const (
	// CacheLayoutDoor leaves the markers to the door's own settings.
	CacheLayoutDoor CacheLayout = iota
	// CacheLayoutNone places no markers of the door's own, keeps the caller's own
	// and turns off a cache the vendor writes on its own where the API allows it.
	CacheLayoutNone
	// CacheLayoutGrowing places the markers of a history that only grows in place
	// of the caller's own, where the door's API takes them: on the system prompt,
	// on the start of the current turn and on a block that moves forward with the
	// history for an hour, and on the last block for five minutes.
	CacheLayoutGrowing
)

// WithCacheLayout sets how the door places the vendor's prompt cache markers.
func WithCacheLayout(layout CacheLayout) CallOption {
	return func(o *CallOptions) {
		o.CacheLayout = layout
	}
}

// WithPromptCacheKey names the requests that share a prompt prefix, such as the
// turns of one conversation, so that a host routing by such a key serves them
// from one cache. A door sends it only to a host that documents the key.
func WithPromptCacheKey(key string) CallOption {
	return func(o *CallOptions) {
		o.PromptCacheKey = key
	}
}

// AddDroppedCacheMarkers records the caller's own cache markers that a growing
// cache layout replaced.
func (w *Warnings) AddDroppedCacheMarkers(model string, dropped int) {
	if dropped == 0 {
		return
	}
	asked := strconv.Itoa(dropped) + " markers"
	if dropped == 1 {
		asked = "1 marker"
	}
	w.Add(Warning{
		Kind: WarningDrop, Option: "WithCacheControl", Model: model, Asked: asked,
		Reason: "the growing cache layout places the request's markers",
	})
}
