package llms

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

// WarningKind names what a door did to a caller option on the way to the wire.
type WarningKind string

const (
	WarningDrop       WarningKind = "drop"
	WarningClamp      WarningKind = "clamp"
	WarningSubstitute WarningKind = "substitute"
	// WarningInherit marks a model the tables do not list: the door applied the
	// rules of the documented release it follows, or sent a request that release
	// refuses.
	WarningInherit WarningKind = "inherit"
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
	{"WithInferenceSpeed", func(o CallOptions) string { return askedString(o.InferenceSpeed) }},
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

// AddInherited records that the tables answer for model with the rules of the
// documented release it follows.
func (w *Warnings) AddInherited(model string) {
	documented, inherited := reasoning.InheritedModel(model)
	if !inherited {
		return
	}
	w.Add(Warning{
		Kind: WarningInherit, Option: "WithModel", Model: model, Asked: model, Sent: model,
		Reason: "the tables do not list this version, so the door applies the rules of " + documented,
	})
}

// KeepRefusal reports whether a door refuses before the network. It does for a
// model the tables list; a model that only inherits the refusal goes out, and the
// refusal is recorded against option with sent, what the door puts on the wire
// instead ("" when nothing).
func (w *Warnings) KeepRefusal(model, option, asked, sent string, refusal error) bool {
	documented, inherited := reasoning.InheritedModel(model)
	if !inherited {
		return true
	}
	w.Add(Warning{
		Kind: WarningInherit, Option: option, Model: model, Asked: asked, Sent: sent,
		Reason: fmt.Sprintf("%s refuses this (%v), but the tables do not list this version", documented, refusal),
	})
	return false
}

// KeepOffRefusal is KeepRefusal for WithReasoningDisabled on a model whose thinking
// cannot be switched off; off is the disable the door sends when the model only
// inherits the refusal.
func (w *Warnings) KeepOffRefusal(model string, off reasoning.OffWire) bool {
	return w.KeepRefusal(model, "WithReasoningDisabled", "off", offSent(off),
		&reasoning.ErrReasoningOffUnsupported{Model: model})
}

func offSent(off reasoning.OffWire) string {
	switch off { //nolint:exhaustive // every other wire is a plain disable
	case reasoning.OffOmit, reasoning.OffUnsupported:
		return ""
	case reasoning.OffMinimalLevel:
		return "minimal"
	case reasoning.OffBetweenToolsClaude:
		return "between_tools"
	}
	return "off"
}

func (w *Warnings) AddOffFloor(model, floor string) {
	w.Add(Warning{
		Kind: WarningSubstitute, Option: "WithReasoningDisabled", Model: model,
		Asked: "off", Sent: floor,
		Reason: "this model has no off switch, only a lowest thinking level",
	})
}

// ClampClaudeTemperature takes the temperature the thinking rules left in place:
// a value they replaced is theirs to report, not a clamp.
func (w *Warnings) ClampClaudeTemperature(model string, temperature *float64) *float64 {
	if temperature == nil {
		return nil
	}
	clamped := reasoning.ClaudeClampTemperature(model, *temperature)
	if clamped == *temperature {
		return temperature
	}
	w.Add(Warning{
		Kind: WarningClamp, Option: "WithTemperature", Model: model,
		Asked: renderFloat(temperature), Sent: renderFloat(&clamped),
		Reason: "Claude takes a temperature from 0 to 1",
	})
	return &clamped
}

func (w *Warnings) List() []Warning {
	if w == nil {
		return nil
	}
	return w.items
}
