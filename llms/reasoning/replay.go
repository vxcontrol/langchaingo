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
	// ThinkingDefault leaves thinking to the vendor's default for the model.
	ThinkingDefault ThinkingMode = iota
	ThinkingOff
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
	// PastReasoningOwnTurns: every earlier answer of the model itself, for the
	// host's cache; another model's answers need none.
	PastReasoningOwnTurns
	PastReasoningEveryTurn
)

// LoopForm is how an unfinished tool loop crosses an epoch boundary.
type LoopForm int

const (
	LoopAsSent LoopForm = iota
	LoopWithoutThinking
	LoopInSummary
)

// ReplayTarget is the model a history goes back to and the way it gets there.
type ReplayTarget struct {
	Model string
	// Host is the lowercase host name of the endpoint, without a port.
	Host  string
	API   ReplayAPI
	Tools bool
	Mode  ThinkingMode
	// KeepsPastReasoning: the caller asks the host to keep the reasoning of
	// earlier turns where the host makes it a switch: Z.ai's and DashScope's
	// clear_thinking false or preserve_thinking true, Moonshot's thinking.keep.
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
}

// ReplayPolicy returns how the history goes back to target.
func ReplayPolicy(target ReplayTarget) Replay {
	r := replayFor(target)
	if documented, inherited := InheritedModel(target.Model); inherited {
		r.Inherits = documented
	}
	return r
}

// NeedsBoundary reports whether target has to start a new epoch before it
// continues a history writer wrote. openLoop is a history that ends in a tool
// loop writer left unfinished.
func NeedsBoundary(target ReplayTarget, writer string, openLoop bool) bool {
	if writer == "" {
		return false
	}
	if replayFor(target).Needs == PastReasoningEveryTurn && vendorFamily(writer) != vendorFamily(target.Model) {
		return true
	}
	return openLoop && claudeOnBudget(target) && claudeRelease(writer) != claudeRelease(target.Model)
}

func (t ReplayTarget) thinks() bool {
	if t.Mode != ThinkingDefault {
		return t.Mode != ThinkingOff
	}
	return !IsClaude(t.Model) || ClaudeThinkingDefaultsOn(t.Model)
}

var asSent = Replay{OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}

func replayFor(t ReplayTarget) Replay {
	switch t.API {
	case ReplayMessages:
		if vendorOfHost(t.Host) != VendorUnknown {
			return vendorReplay(t)
		}
		if IsClaude(t.Model) {
			return claudeOverMessages(t)
		}
	case ReplayConverse:
		if IsClaude(t.Model) {
			return claudeOnConverse(t.Model, t.Tools)
		}
	case ReplayChat:
		if !IsClaude(t.Model) {
			return vendorReplay(t)
		}
		switch {
		case PublicProviderHost(t.Host):
			return asSent
		case bedrockConverseRoute(t.Model):
			return claudeOnConverse(t.Model, t.Tools)
		}
		r := claudeReplay(t.Model, t.Tools)
		r.OwnLoop, r.ForeignLoop = LoopAsSent, LoopInSummary
		if r.ChecksPrefix {
			r.Binding, r.OwnLoop = BindingPrefix, LoopWithoutThinking
		}
		return r
	case ReplayGemini:
		if GeminiUsesThinkingLevel(t.Model) {
			return geminiReplay(t.Model, t.Tools)
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

func claudeOverMessages(t ReplayTarget) Replay {
	r := claudeReplay(t.Model, t.Tools)
	switch {
	case r.ChecksPrefix:
		r.Binding, r.OwnLoop, r.ForeignLoop = BindingPrefix, LoopWithoutThinking, LoopWithoutThinking
	case claudeOnBudget(t):
		r.OwnLoop, r.ForeignLoop = LoopAsSent, LoopInSummary
	default:
		r.OwnLoop, r.ForeignLoop = LoopAsSent, LoopWithoutThinking
	}
	return r
}

func claudeOnConverse(model string, tools bool) Replay {
	r := claudeReplay(model, tools)
	r.Binding, r.OwnLoop, r.ForeignLoop = BindingAllMessages, LoopInSummary, LoopInSummary
	return r
}

func claudeOnBudget(t ReplayTarget) bool {
	return IsClaude(t.Model) && t.thinks() && !ResolveClaudeAdaptive(t.Model, t.Mode != ThinkingBudget)
}

func bedrockConverseRoute(model string) bool {
	route := strings.ToLower(model)
	return strings.HasPrefix(route, "bedrock/") && !strings.HasPrefix(route, "bedrock/invoke/")
}

func geminiReplay(model string, tools bool) Replay {
	r := Replay{Binding: BindingCurrentTurn, OwnLoop: LoopWithoutThinking, ForeignLoop: LoopWithoutThinking}
	switch {
	case geminiFrom(model, 3, 5):
		r.Needs = PastReasoningEveryTurn
	case tools:
		r.Needs, r.HostDropsPast = PastReasoningOpenLoop, true
	default:
		r.HostDropsPast = true
	}
	return r
}

func geminiFrom(model string, major, minor int) bool {
	asked, askedMinor, ok := generationAfter("gemini-", routedName(model))
	return ok && (asked > major || asked == major && askedMinor >= minor)
}

var (
	everyTurn = Replay{Needs: PastReasoningEveryTurn, OwnLoop: LoopAsSent, ForeignLoop: LoopInSummary}
	dropsPast = Replay{HostDropsPast: true, OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}
)

func vendorReplay(t ReplayTarget) Replay {
	switch ServedBy(t.Model, t.Host) { //nolint:exhaustive // the other vendors replay as sent
	case VendorDeepSeek:
		if ServedByDeepSeek(t.Model, t.Host) && t.thinks() {
			if t.Tools {
				return everyTurn
			}
			return dropsPast
		}
	case VendorMoonshot:
		switch {
		case namesGeneration(t.Model, "kimi-k3"), namesGeneration(t.Model, "kimi-k2.7-code"):
			return everyTurn
		case namesGeneration(t.Model, "kimi-k2.6"):
			return keptOnRequest(t)
		}
	case VendorMiniMax:
		if t.Tools && namesMiniMax(t.Model, "minimax-m") {
			return everyTurn
		}
	case VendorDashScope:
		return dashScopeReplay(t)
	case VendorZAI:
		return keptOnRequest(t)
	case VendorMistral:
		if t.thinks() && ReplaysThinkingInContent(t.Model) {
			return everyTurn
		}
	case VendorXAI:
		if GrokFamily(t.Model) && IsReasoningModel(t.Model) {
			return Replay{Needs: PastReasoningOwnTurns, OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}
		}
	}
	return asSent
}

func keptOnRequest(t ReplayTarget) Replay {
	switch {
	case t.KeepsPastReasoning:
		return everyTurn
	case !t.thinks():
		return asSent
	}
	r := dropsPast
	if t.Tools {
		r.Needs = PastReasoningOpenLoop
	}
	return r
}

var (
	dashScopeKeepsByDefault = []string{
		"qwen3.8-max", "qwen3.8-flash", "qwen3.8-omni-flash", "kimi-k2.7-code", "glm-5.2", "glm-5.1", "glm-5", "glm-4.7",
	}
	dashScopeKeepsWhenAsked = []string{
		"qwen3.7-max", "qwen3.7-plus", "qwen3.7-flash", "qwen3.6-max", "qwen3.6-plus", "kimi-k2.6",
		"glm-5.3",
	}
)

func dashScopeReplay(t ReplayTarget) Replay {
	switch {
	case namesAnyGeneration(t.Model, dashScopeKeepsWhenAsked):
		switch {
		case t.KeepsPastReasoning:
			return everyTurn
		case !t.thinks():
			return asSent
		}
		return dropsPast
	case namesAnyGeneration(t.Model, dashScopeKeepsByDefault):
		return everyTurn
	}
	return asSent
}

func namesGeneration(model, generation string) bool {
	for _, form := range modelSpellings(model) {
		if hasGeneration(form, generation) {
			return true
		}
	}
	return false
}

func namesAnyGeneration(model string, generations []string) bool {
	for _, generation := range generations {
		if namesGeneration(model, generation) {
			return true
		}
	}
	return false
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
	case IsGemini(model):
		return "gemini"
	case ServedByMistral(model):
		return "mistral"
	}
	for _, form := range modelSpellings(model) {
		for _, family := range []string{"deepseek", "kimi-", "minimax-", "qwen", "glm-"} {
			if strings.HasPrefix(form, family) {
				return family
			}
		}
	}
	return ""
}
