package llms

import "fmt"

// WarningKind names what a door did to a caller option on the way to the wire.
type WarningKind string

const (
	WarningDrop       WarningKind = "drop"
	WarningClamp      WarningKind = "clamp"
	WarningSubstitute WarningKind = "substitute"
)

// Warning reports one caller option that did not reach the vendor as asked.
// Asked and Sent are rendered values; an empty Sent means nothing reached the wire.
type Warning struct {
	Kind   WarningKind
	Option string
	Model  string
	Asked  string
	Sent   string
	Reason string
}

func (w Warning) String() string {
	sent := w.Sent
	if sent == "" {
		sent = "nothing"
	}
	return fmt.Sprintf("%s: %s asked %s, %s sent %s on %s (%s)",
		w.Kind, w.Option, w.Asked, w.Option, sent, w.Model, w.Reason)
}

// Warnings collects Warning values while a request is built. A nil *Warnings is
// valid and keeps nothing.
type Warnings struct {
	items []Warning
}

func (w *Warnings) Add(warning Warning) {
	if w == nil {
		return
	}
	w.items = append(w.items, warning)
}

func (w *Warnings) List() []Warning {
	if w == nil {
		return nil
	}
	return w.items
}
