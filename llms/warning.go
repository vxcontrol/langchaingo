package llms

import (
	"fmt"
	"strconv"
)

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

func renderFloat(v *float64) string {
	if v == nil {
		return ""
	}
	return strconv.FormatFloat(*v, 'g', -1, 64)
}

func renderInt(v *int) string {
	if v == nil {
		return ""
	}
	return strconv.Itoa(*v)
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

// AddFloatChange records an optional float option that the door changed on the
// way to the wire. A nil after means the option never reached it.
func (w *Warnings) AddFloatChange(option, model, reason string, before, after *float64) {
	w.addChange(option, model, reason, renderFloat(before), renderFloat(after))
}

func (w *Warnings) AddIntChange(option, model, reason string, before, after *int) {
	w.addChange(option, model, reason, renderInt(before), renderInt(after))
}

func (w *Warnings) addChange(option, model, reason, before, after string) {
	switch {
	case before == "" || before == after:
		return
	case after == "":
		w.Add(Warning{
			Kind: WarningDrop, Option: option, Model: model,
			Asked: before, Reason: reason,
		})
	default:
		w.Add(Warning{
			Kind: WarningSubstitute, Option: option, Model: model,
			Asked: before, Sent: after, Reason: reason,
		})
	}
}

func (w *Warnings) List() []Warning {
	if w == nil {
		return nil
	}
	return w.items
}
