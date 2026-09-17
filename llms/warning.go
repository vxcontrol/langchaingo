package llms

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
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

var unreadCatalogue = []struct {
	option string
	asked  func(CallOptions) string
}{
	{"WithMinP", func(o CallOptions) string { return askedFloat(o.MinP) }},
	{"WithRepetitionPenalty", func(o CallOptions) string { return askedFloat(o.RepetitionPenalty) }},
	{"WithFrequencyPenalty", func(o CallOptions) string { return askedFloat(o.FrequencyPenalty) }},
	{"WithPresencePenalty", func(o CallOptions) string { return askedFloat(o.PresencePenalty) }},
	{"WithTopK", func(o CallOptions) string { return askedInt(o.TopK, 0) }},
	{"WithN", func(o CallOptions) string { return askedInt(o.N, 1) }},
	{"WithCandidateCount", func(o CallOptions) string { return askedInt(o.CandidateCount, 1) }},
	{"WithTopLogProbs", func(o CallOptions) string { return askedInt(o.TopLogProbs, 0) }},
	{"WithMinLength", func(o CallOptions) string { return askedInt(o.MinLength, 0) }},
	{"WithMaxLength", func(o CallOptions) string { return askedInt(o.MaxLength, 0) }},
	{"WithSeed", func(o CallOptions) string {
		if o.Seed == nil {
			return ""
		}
		return strconv.Itoa(*o.Seed)
	}},
	{"WithVerbosity", func(o CallOptions) string { return askedString(o.Verbosity) }},
	{"WithResponseMIMEType", func(o CallOptions) string { return askedString(o.ResponseMIMEType) }},
	{"WithLogProbs", func(o CallOptions) string {
		if o.LogProbs != nil && *o.LogProbs {
			return "true"
		}
		return ""
	}},
	{"WithJSONMode", func(o CallOptions) string {
		if o.JSONMode && o.StructuredOutput == nil {
			return "true"
		}
		return ""
	}},
}

func askedFloat(v *float64) string {
	if v == nil || *v == 0 {
		return ""
	}
	return strconv.FormatFloat(*v, 'g', -1, 64)
}

func askedInt(v *int, neutral int) string {
	if v == nil || *v == neutral {
		return ""
	}
	return strconv.Itoa(*v)
}

func askedString(v *string) string {
	if v == nil {
		return ""
	}
	return *v
}

// AddUnreadOptions reports every catalogue option the caller set that this door
// leaves off the wire. carried names the ones it does put there.
func (w *Warnings) AddUnreadOptions(model string, opts CallOptions, reason string, carried ...string) {
	onTheWire := make(map[string]bool, len(carried))
	for _, option := range carried {
		onTheWire[option] = true
	}
	for _, entry := range unreadCatalogue {
		if onTheWire[entry.option] {
			continue
		}
		if asked := entry.asked(opts); asked != "" {
			w.Add(Warning{
				Kind: WarningDrop, Option: entry.option, Model: model,
				Asked: asked, Reason: reason,
			})
		}
	}
}

// AddUnreadExtraBody records the WithExtraBody fields a door cannot merge.
func (w *Warnings) AddUnreadExtraBody(model string, opts CallOptions, reason string) {
	extraBody := ExtraBody(opts)
	if len(extraBody) == 0 {
		return
	}
	keys := make([]string, 0, len(extraBody))
	for key := range extraBody {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	w.Add(Warning{
		Kind: WarningDrop, Option: "WithExtraBody", Model: model,
		Asked: strings.Join(keys, ", "), Reason: reason,
	})
}

func (w *Warnings) List() []Warning {
	if w == nil {
		return nil
	}
	return w.items
}
