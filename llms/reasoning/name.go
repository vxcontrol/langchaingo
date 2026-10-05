package reasoning

import "strings"

// modelSpellings returns the forms a capability rule may match against. Every form
// added here widens the rules that read it, so a new one must never let a rule match
// a model the bare name would not. A version missing from lines.go arrives as the
// release it follows, so a table row keyed on that version never fires until it is listed there.
func modelSpellings(model string) []string {
	vendor, bare := splitModelName(model)
	if bare == "" {
		return nil
	}
	if earlier, ok := earlierNames[bare]; ok && vendor == "" {
		return []string{earlier, bare}
	}
	bare = inheritedOrSelf(bare)
	if vendor == "" {
		return []string{bare}
	}
	return []string{bare, vendor + "-" + bare}
}

func splitModelName(model string) (vendor, bare string) {
	m := routedName(model)
	m = stripFineTuneWrapper(m)
	m = stripBedrockRegion(m)

	vendor, bare = splitPlatformPrefix(m)
	bare = documentedAlias(bare)
	if vendor == "" {
		if dashVendor, stripped, ok := stripDashWrittenVendor(bare); ok {
			return dashVendor, stripped
		}
	}
	return vendor, bare
}

func routedName(model string) string {
	m := strings.ToLower(model)
	idx := strings.LastIndex(m, "/")
	if idx == -1 {
		return m
	}
	route, name := m[:idx], m[idx+1:]
	if strings.HasPrefix(route[strings.LastIndex(route, "/")+1:], "~") {
		return routerAlias(name)
	}
	return name
}

// vendorAliases entries must be backed by a vendor page that serves both names with one model.
var vendorAliases = map[string]string{
	"gemini-pro-latest": "gemini-3.1-pro-preview",
	"zai-glm-5":         "zai-glm-5-3", "zai-glm-latest": "zai-glm-5-3",
	"qwen-plus-latest": "qwen-plus", "qwen-turbo-latest": "qwen-turbo",
	"qwen3-max-2026-01-23": "qwen3-max", "qwen3-max-preview": "qwen3-max",
	"qwen3-vl-plus-2025-12-19": "qwen3-vl-plus",
}

// qwenSnapshotsFrom entries come from Model Studio's deep-thinking page: the first
// dated snapshot it lists with the stable id's thinking.
var qwenSnapshotsFrom = map[string]string{"qwen-plus": "2025-04-28", "qwen-flash": "2025-07-28"}

func documentedAlias(bare string) string {
	if alias, ok := vendorAliases[bare]; ok {
		return alias
	}
	for stable, first := range qwenSnapshotsFrom {
		date, ok := strings.CutPrefix(bare, stable+"-")
		if ok && len(date) == len(first) && date[4] == '-' && date >= first {
			return stable
		}
	}
	return bare
}

func inheritedOrSelf(bare string) string {
	if documented, ok := inheritLine(bare); ok {
		return documented
	}
	return bare
}

// dashWrittenVendors are platform prefixes a vendor also publishes with a dash
// where splitPlatformPrefix only reads a dot. Adding one here must be backed by
// a vendor listing that names both spellings as one model.
var dashWrittenVendors = []string{"zai"}

// earlierNames entries must be backed by a vendor page that serves both names with one model.
var earlierNames = map[string]string{"deepseek-flash": "deepseek-v4-flash"}

func stripDashWrittenVendor(model string) (vendor, rest string, ok bool) {
	for _, vendor := range dashWrittenVendors {
		if rest, ok := strings.CutPrefix(model, vendor+"-"); ok && rest != "" {
			return vendor, rest, true
		}
	}
	return "", "", false
}

func stripBedrockRegion(model string) string {
	for _, prefix := range bedrockRegionPrefixes {
		if rest, ok := strings.CutPrefix(model, prefix); ok && rest != "" {
			return rest
		}
	}
	return model
}

func splitPlatformPrefix(model string) (vendor, rest string) {
	rest = model
	for {
		idx := strings.Index(rest, ".")
		if idx <= 0 || !isAlpha(rest[:idx]) {
			return vendor, rest
		}
		vendor, rest = rest[:idx], rest[idx+1:]
	}
}

// hasGeneration rejects a trailing digit, so it must not be used where the digits
// that follow are a date rather than a later generation.
func hasGeneration(model, generation string) bool {
	rest, ok := strings.CutPrefix(model, generation)
	if !ok {
		return false
	}
	return rest == "" || rest[0] < '0' || rest[0] > '9'
}

func stripFineTuneWrapper(model string) string {
	rest, ok := strings.CutPrefix(model, "ft:")
	if !ok {
		return model
	}
	base, _, _ := strings.Cut(rest, ":")
	if base == "" {
		return model
	}

	return base
}
