package reasoning

import "strings"

// modelSpellings returns the forms a capability rule may match against. Every form
// added here widens the rules that read it, so a new one must never let a rule match
// a model the bare name would not.
func modelSpellings(model string) []string {
	m := strings.ToLower(model)
	if idx := strings.LastIndex(m, "/"); idx != -1 {
		m = m[idx+1:]
	}
	m = stripFineTuneWrapper(m)
	m = stripBedrockRegion(m)

	vendor, bare := splitPlatformPrefix(m)
	switch {
	case bare == "":
		return nil
	case vendor == "":
		if stripped, ok := stripDashWrittenVendor(bare); ok {
			return []string{stripped, bare}
		}
		if earlier, ok := earlierNames[bare]; ok {
			return []string{earlier, bare}
		}
		return []string{bare}
	default:
		return []string{bare, vendor + "-" + bare}
	}
}

// dashWrittenVendors are platform prefixes a vendor also publishes with a dash
// where splitPlatformPrefix only reads a dot. Adding one here must be backed by
// a vendor listing that names both spellings as one model.
var dashWrittenVendors = []string{"zai"}

// earlierNames entries must be backed by a vendor page that serves both names with one model.
var earlierNames = map[string]string{"deepseek-flash": "deepseek-v4-flash"}

func stripDashWrittenVendor(model string) (string, bool) {
	for _, vendor := range dashWrittenVendors {
		if rest, ok := strings.CutPrefix(model, vendor+"-"); ok && rest != "" {
			return rest, true
		}
	}
	return "", false
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
