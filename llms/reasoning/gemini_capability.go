package reasoning

import (
	"regexp"
	"strconv"
	"strings"
)

// This file is the single source of truth for Google (Gemini / Gemma) reasoning
// classification, so the enable path, the disable path, the reasoning-model
// detector, and the UI hint all agree instead of re-deriving it from scattered
// model-string checks. Provider-wire specifics that need the genai types
// (thinking_level mapping, the temperature value) stay in the googleai adapter.

func baseModelName(model string) string {
	m := strings.ToLower(model)
	if idx := strings.LastIndex(m, "/"); idx != -1 {
		m = m[idx+1:]
	}
	return m
}

var geminiVersionPattern = regexp.MustCompile(`^gemini-(\d+)(?:\.\d+)?(?:-|$)`)

func geminiMajor(model string) (int, bool) {
	m := geminiVersionPattern.FindStringSubmatch(model)
	if m == nil {
		return 0, false
	}
	major, err := strconv.Atoi(m[1])
	return major, err == nil
}

func geminiLatestAlias(model string) bool {
	return strings.HasPrefix(model, "gemini-") && strings.HasSuffix(model, "-latest")
}

func GeminiTakesNoCandidateCount(model string) bool {
	m := baseModelName(model)
	if geminiLatestAlias(m) {
		return true
	}
	major, ok := geminiMajor(m)
	return ok && major >= 3
}

func hasFamily(model, family string) bool {
	rest := model
	for {
		idx := strings.Index(rest, family)
		if idx == -1 {
			return false
		}
		rest = rest[idx+len(family):]
		if rest == "" || rest[0] < '0' || rest[0] > '9' {
			return true
		}
	}
}

// GeminiSupportsThinking reports whether the model belongs to a Google thinking
// family: Gemini 2.5, Gemini 3.x, or Gemma 4.
func GeminiSupportsThinking(model string) bool {
	m := baseModelName(model)
	if geminiNonChatSurface(m) {
		return false
	}
	return hasFamily(m, "gemini-2.5") ||
		hasFamily(m, "gemini-3") ||
		hasFamily(m, "gemma-4") ||
		geminiUnversionedThinking(m)
}

func geminiNonChatSurface(model string) bool {
	return strings.Contains(model, "-tts") ||
		strings.Contains(model, "-live-translate") ||
		(strings.Contains(model, "-image") && !hasFamily(model, "gemini-3")) ||
		strings.Contains(model, "transcribe")
}

func geminiUnversionedThinking(model string) bool {
	for _, prefix := range []string{"gemini-flash-latest", "gemini-flash-lite-latest", "gemini-robotics-er"} {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// GeminiUsesThinkingLevel reports whether the model uses the qualitative
// thinking_level control (Gemini 3.x), where thinking_budget is deprecated,
// instead of a token budget. Gemini 3 also recommends running at temperature 1.0.
func GeminiUsesThinkingLevel(model string) bool {
	return hasFamily(baseModelName(model), "gemini-3")
}

// GeminiThinkingLevels returns the thinking levels Google documents for a model
// that takes fewer than all of them: an empty set means the model thinks but no
// level is documented, nil means no narrower set is recorded.
func GeminiThinkingLevels(model string) []string {
	m := baseModelName(model)
	image := strings.Contains(m, "-image")
	switch {
	case image && hasFamily(m, "gemini-3.1-flash"):
		return []string{"minimal", "high"}
	case image && hasFamily(m, "gemini-3-pro"):
		return []string{}
	case hasFamily(m, "gemini-3-pro"):
		return []string{"low", "high"}
	}
	return nil
}

// GeminiTakesNoThinkingBudget reports whether Google documents only thinking
// levels for the model, so an explicit budget has to travel as a level.
func GeminiTakesNoThinkingBudget(model string) bool {
	m := baseModelName(model)
	return strings.Contains(m, "-image") && hasFamily(m, "gemini-3")
}

// GeminiAcceptsMinimalLevel reports whether the model takes thinking_level
// MINIMAL. A name this package has not measured reports false and falls back
// to LOW, so extending this set means measuring first, not guessing.
func GeminiAcceptsMinimalLevel(model string) bool {
	m := baseModelName(model)
	if !GeminiUsesThinkingLevel(m) || strings.Contains(m, "pro") {
		return false
	}
	for _, family := range []string{"gemini-3.1", "gemini-3.5", "gemini-3.6"} {
		if hasFamily(m, family) {
			return true
		}
	}
	return !strings.HasPrefix(m, "gemini-3.")
}

// geminiKnownNonThinking reports whether the model is a pre-thinking Gemini/Gemma
// generation that never thinks (Gemini 1.x/2.0, Gemma 1–3), so it takes no thinking
// control at all. Unclassified names are NOT matched, staying optimistic so a
// future thinking model is not wrongly treated as non-thinking.
func geminiKnownNonThinking(model string) bool {
	m := baseModelName(model)
	return hasFamily(m, "gemini-1") ||
		hasFamily(m, "gemini-2.0") ||
		hasFamily(m, "gemma-1") ||
		hasFamily(m, "gemma-2") ||
		hasFamily(m, "gemma-3")
}

// GeminiTogglesThinkingByLevel reports whether the model expresses thinking as
// on or off through thinking_level alone, with no budget and no level between.
// Kept apart from GeminiUsesThinkingLevel, which also turns on the Gemini 3
// thought-signature placeholder.
func GeminiTogglesThinkingByLevel(model string) bool {
	return hasFamily(baseModelName(model), "gemma-4")
}

// GeminiCanDisable reports whether thinking can be turned off at all; ResolveOff
// decides the wire. Unclassified Google models are treated as disablable, so a
// model this package has not seen is attempted rather than refused.
func GeminiCanDisable(model string) bool {
	m := baseModelName(model)
	if GeminiThinkingOffByDefault(m) {
		return true
	}
	if hasFamily(m, "gemini-3") {
		return GeminiAcceptsMinimalLevel(m)
	}
	if hasFamily(m, "gemini-2.5") && strings.Contains(m, "pro") {
		return false
	}
	return true
}

// GeminiThinkingOffByDefault reports whether the model leaves thinking off until
// asked, so omitting the thinking config already yields "off".
func GeminiThinkingOffByDefault(model string) bool {
	return hasFamily(baseModelName(model), "gemini-2.5") &&
		strings.Contains(baseModelName(model), "flash-lite")
}

// GeminiBudgetRange returns the thinking_budget bounds the vendor documents for
// the model. An unknown model reports no bounds.
func GeminiBudgetRange(model string) (minimum, maximum int, known bool) {
	m := baseModelName(model)
	if !hasFamily(m, "gemini-2.5") {
		return 0, 0, false
	}
	switch {
	case strings.Contains(m, "pro"):
		return 128, 32768, true
	case strings.Contains(m, "flash-lite"):
		return 512, 24576, true
	case strings.Contains(m, "flash"):
		return 1, 24576, true
	default:
		return 0, 0, false
	}
}
