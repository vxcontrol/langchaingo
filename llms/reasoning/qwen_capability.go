package reasoning

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

var qwenParameterCountName = regexp.MustCompile(`^qwen3-\d+(\.\d+)?b(-a\d+(\.\d+)?b)?$`)

var qwenDefaultThinkingName = regexp.MustCompile(`^qwen3\.[567]-`)

var qwenNextName = regexp.MustCompile(`^qwen3-next-.*-thinking$`)

var qwenVLOpenWeightName = regexp.MustCompile(`^qwen3-vl-\d+(\.\d+)?b(-a\d+(\.\d+)?b)?$`)

// QwenThinkingOffByFlag reports whether the model thinks until enable_thinking
// false asks it to stop.
func QwenThinkingOffByFlag(model string) bool {
	return qwenDefaultThinkingName.MatchString(dashScopeSpelling(model))
}

// QwenThinkingRequiresStream reports whether DashScope serves the model's
// thinking only on a streaming call.
func QwenThinkingRequiresStream(model string) bool {
	return qwenParameterCountName.MatchString(dashScopeSpelling(model))
}

func dashScopeSpelling(model string) string {
	m := strings.TrimPrefix(strings.ToLower(model), "dashscope/")
	if strings.Contains(m, "/") {
		return ""
	}
	return m
}

// ErrThinkingRequiresStream reports thinking asked of a stream-only model
// without a stream.
type ErrThinkingRequiresStream struct{ Model string }

func (e *ErrThinkingRequiresStream) Error() string {
	return fmt.Sprintf("reasoning on model %q is available only on a streaming call", e.Model)
}

var qwenFlagThinkers = map[string]bool{
	"qwen-plus": true, "qwen-flash": true, "qwen-turbo": true,
	"qwen3-max": true, "qwen3-vl-plus": true, "qwen3-vl-flash": true,
}

// QwenThinkingEnabledByFlag reports whether DashScope leaves the model's
// thinking off until enable_thinking:true asks for it.
func QwenThinkingEnabledByFlag(model string) bool {
	return qwenFlagThinkers[dashScopeSpelling(model)]
}

// DashScopeGuestThinkingEnabledByFlag reports whether a guest model DashScope
// serves leaves its thinking off until enable_thinking:true asks for it.
func DashScopeGuestThinkingEnabledByFlag(model string) bool {
	guest, ok := strings.CutPrefix(strings.ToLower(model), "dashscope/")
	return ok && (guest == "kimi-k2.6" || guest == "kimi-k2.5")
}

var dashScopeGuestBudget = []string{
	"glm-5.2", "glm-5.1", "glm-4.7", "glm-4.6", "glm-4.5",
	"kimi-k2.5", "kimi-k2.6", "kimi-k2.7-code", "kimi-k2-thinking",
}

var dashScopeDeepSeekBudget = []string{"deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-0731"}

func dashScopeGuestSpelling(model string) string {
	rest, ok := strings.CutPrefix(strings.ToLower(model), "dashscope/")
	rest = strings.TrimPrefix(rest, "kimi/")
	if !ok || strings.Contains(rest, "/") {
		return ""
	}
	return rest
}

var dashScopeHosts = []string{
	"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com",
	"cn-hongkong.dashscope.aliyuncs.com",
}

// DashScopeRoute returns the name the rules read for a model sent to host: a
// name sent to Model Studio's own host reads as its dashscope/ route.
func DashScopeRoute(model, host string) string {
	onDashScope := slices.Contains(dashScopeHosts, host) || strings.HasSuffix(host, ".maas.aliyuncs.com")
	if !onDashScope || strings.HasPrefix(strings.ToLower(model), "dashscope/") {
		return model
	}
	return "dashscope/" + model
}

// DashScopeTakesNoTopK reports whether DashScope serves the model from a family
// that takes no top_k.
func DashScopeTakesNoTopK(model string) bool {
	rest, ok := strings.CutPrefix(strings.ToLower(model), "dashscope/")
	name := rest[strings.LastIndex(rest, "/")+1:]
	return ok && (strings.HasPrefix(name, "deepseek") || strings.HasPrefix(name, "kimi-") ||
		strings.HasPrefix(name, "moonshot-kimi") || strings.HasPrefix(name, "minimax-"))
}

// DashScopeTakesThinkingBudget reports whether DashScope caps the model's
// thinking by a token budget.
func DashScopeTakesThinkingBudget(model string) bool {
	if guest := dashScopeGuestSpelling(model); guest != "" {
		if guest == "glm-5" || slices.Contains(dashScopeDeepSeekBudget, guest) {
			return true
		}
		for _, generation := range dashScopeGuestBudget {
			if hasGeneration(guest, generation) {
				return true
			}
		}
	}
	m := dashScopeSpelling(model)
	if m == "" {
		return false
	}
	return qwenDefaultThinkingName.MatchString(m) ||
		qwenParameterCountName.MatchString(m) ||
		qwenNextName.MatchString(m) ||
		qwenVLOpenWeightName.MatchString(m) ||
		qwenFlagThinkers[m] ||
		hasGeneration(m, "qwen3.8")
}

// DashScopeBudgetSharesAnswerLimit reports whether the model's thinking budget
// counts against max_tokens together with the answer.
func DashScopeBudgetSharesAnswerLimit(model string) bool {
	return slices.Contains(dashScopeDeepSeekBudget, dashScopeGuestSpelling(model))
}
