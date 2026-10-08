package reasoning

import "strings"

// ReplayAPI is the request format a door sends the history in.
type ReplayAPI int

const (
	// ReplayMessages is Anthropic's Messages API: the Claude API, Google Cloud, a gateway's passthrough.
	ReplayMessages ReplayAPI = iota
	ReplayConverse
	// ReplayChat is Chat Completions, on a vendor's own API or on a gateway.
	ReplayChat
	ReplayGemini
	ReplayOllama
)

// ThinkingMode is how the caller asks the model to think.
type ThinkingMode int

const (
	ThinkingOff ThinkingMode = iota
	ThinkingBudget
	ThinkingAdaptive
)

// Binding is what a replayed reasoning block stays valid against.
type Binding int

const (
	BindingNone Binding = iota
	// BindingPrefix: the system prompt, the tools and every message before the block.
	BindingPrefix
	// BindingAllMessages: every message of the conversation.
	BindingAllMessages
	// BindingCurrentTurn: the function calls of the current turn carry signatures.
	BindingCurrentTurn
)

// PastReasoning is the reasoning of earlier answers a model needs back.
type PastReasoning int

const (
	PastReasoningUnused PastReasoning = iota
	// PastReasoningOpenLoop: the answers of an unfinished tool loop.
	PastReasoningOpenLoop
	PastReasoningEveryTurn
)

// LoopForm is how an unfinished tool loop crosses an epoch boundary.
type LoopForm int

const (
	LoopAsSent LoopForm = iota
	LoopWithoutThinking
	LoopInSummary
)

// ReplayOptions are switches a caller sends the host about earlier reasoning.
type ReplayOptions struct {
	// KeepsPastReasoning: the caller asks the host to keep the reasoning of
	// earlier turns where the host makes it a switch (Z.ai, Kimi K2.6).
	KeepsPastReasoning bool
}

// Replay is how a history goes back to a model.
type Replay struct {
	Binding      Binding
	ChecksPrefix bool
	Needs        PastReasoning
	// HostDropsPast: the host leaves the reasoning of earlier turns out of the
	// context on its own.
	HostDropsPast bool
	// OwnLoop and ForeignLoop are the forms of an unfinished tool loop the model
	// wrote itself and another model wrote.
	OwnLoop, ForeignLoop LoopForm
	// Inherits is the listed release whose rules a version the tables do not
	// list follows, or "".
	Inherits string

	reader readerOf
}

type readerOf struct {
	model    string
	family   string
	onBudget bool
}

// ReplayPolicy returns how the history goes back to model on host through api.
func ReplayPolicy(model, host string, api ReplayAPI, tools bool, mode ThinkingMode, opts ReplayOptions) Replay {
	r := replayFor(model, host, api, tools, mode, opts)
	if documented, inherited := InheritedModel(model); inherited {
		r.Inherits = documented
	}
	r.reader = readerOf{model: model, family: vendorFamily(model), onBudget: claudeOnBudget(model, mode)}
	return r
}

func replayFor(model, host string, api ReplayAPI, tools bool, mode ThinkingMode, opts ReplayOptions) Replay {
	asSent := Replay{OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}
	switch api {
	case ReplayMessages:
		if vendorOfHost(host) != VendorUnknown {
			return vendorReplay(model, host, tools, opts)
		}
		if IsClaude(model) {
			return claudeOverMessages(model, tools, mode)
		}
	case ReplayConverse:
		if IsClaude(model) {
			r := claudeReplay(model, tools)
			r.Binding, r.OwnLoop, r.ForeignLoop = BindingAllMessages, LoopInSummary, LoopInSummary
			return r
		}
	case ReplayChat:
		if IsClaude(model) {
			if PublicProviderHost(host) {
				return asSent
			}
			r := claudeReplay(model, tools)
			r.OwnLoop, r.ForeignLoop = LoopAsSent, LoopInSummary
			if r.ChecksPrefix {
				r.Binding, r.OwnLoop = BindingPrefix, LoopWithoutThinking
			}
			return r
		}
		return vendorReplay(model, host, tools, opts)
	case ReplayGemini:
		if GeminiUsesThinkingLevel(model) {
			r := Replay{Binding: BindingCurrentTurn, OwnLoop: LoopWithoutThinking, ForeignLoop: LoopWithoutThinking}
			if tools {
				r.Needs = PastReasoningOpenLoop
			}
			return r
		}
	case ReplayOllama:
	}
	return asSent
}

var (
	prefixCheckingClaude  = []string{"claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"}
	keepAllThinkingClaude = []string{
		"claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
		"claude-sonnet-4-6", "claude-sonnet-5", "claude-haiku-5-5",
		"claude-fable-5", "claude-mythos-5", "claude-mythos-preview",
	}
)

func claudeReplay(model string, tools bool) Replay {
	r := Replay{
		ChecksPrefix:  claudeNamedIn(model, prefixCheckingClaude),
		HostDropsPast: !claudeNamedIn(model, keepAllThinkingClaude),
	}
	if tools {
		r.Needs = PastReasoningOpenLoop
	}
	return r
}

func claudeOverMessages(model string, tools bool, mode ThinkingMode) Replay {
	r := claudeReplay(model, tools)
	switch {
	case r.ChecksPrefix:
		r.Binding, r.OwnLoop, r.ForeignLoop = BindingPrefix, LoopWithoutThinking, LoopWithoutThinking
	case claudeOnBudget(model, mode):
		r.OwnLoop, r.ForeignLoop = LoopAsSent, LoopInSummary
	default:
		r.OwnLoop, r.ForeignLoop = LoopAsSent, LoopWithoutThinking
	}
	return r
}

func claudeOnBudget(model string, mode ThinkingMode) bool {
	return IsClaude(model) && mode != ThinkingOff && !ResolveClaudeAdaptive(model, mode == ThinkingAdaptive)
}

func vendorReplay(model, host string, tools bool, opts ReplayOptions) Replay {
	everyTurn := Replay{Needs: PastReasoningEveryTurn, OwnLoop: LoopAsSent, ForeignLoop: LoopInSummary}
	dropsPast := Replay{HostDropsPast: true, OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}
	switch ServedBy(model, host) { //nolint:exhaustive // the other vendors replay as sent
	case VendorDeepSeek:
		if !ServedByDeepSeek(model, host) {
			break
		}
		if tools {
			return everyTurn
		}
		return dropsPast
	case VendorMoonshot:
		switch {
		case namesGeneration(model, "kimi-k3"), namesGeneration(model, "kimi-k2.7-code"):
			return everyTurn
		case namesGeneration(model, "kimi-k2.6"):
			if opts.KeepsPastReasoning {
				return everyTurn
			}
			return dropsPast
		}
	case VendorMiniMax:
		if tools && namesMiniMax(model, "minimax-m") {
			return everyTurn
		}
	case VendorDashScope:
		if namesFamily(model, "qwen3.8") {
			return everyTurn
		}
	case VendorZAI:
		if !ServedByZAI(model, host) {
			break
		}
		if opts.KeepsPastReasoning {
			return everyTurn
		}
		return dropsPast
	case VendorMistral:
		if ReplaysThinkingInContent(model) {
			return everyTurn
		}
	}
	return Replay{OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}
}

func namesGeneration(model, generation string) bool {
	for _, form := range modelSpellings(model) {
		if hasGeneration(form, generation) {
			return true
		}
	}
	return false
}

// NeedsBoundaryAfter reports whether the model a policy was made for has to
// start a new epoch before it continues a history writer wrote. openLoop is a
// history that ends in a tool loop writer left unfinished.
func (r Replay) NeedsBoundaryAfter(writer string, openLoop bool) bool {
	if writer == "" {
		return false
	}
	if r.Needs == PastReasoningEveryTurn && vendorFamily(writer) != r.reader.family {
		return true
	}
	return openLoop && r.reader.onBudget && claudeRelease(writer) != claudeRelease(r.reader.model)
}

func claudeRelease(model string) string {
	if _, id, ok := claudeID(claudeName(model)); ok {
		if release, found := documentedClaude(id); found {
			return release
		}
	}
	return model
}

func vendorFamily(model string) string {
	switch {
	case IsClaude(model):
		return "claude"
	case IsGemini(model):
		return "gemini"
	case ServedByMistral(model):
		return "mistral"
	}
	for _, form := range modelSpellings(model) {
		for _, family := range []string{"deepseek", "kimi-", "minimax-", "qwen", "glm-", "grok-", "gpt-"} {
			if strings.HasPrefix(form, family) {
				return family
			}
		}
	}
	return ""
}
