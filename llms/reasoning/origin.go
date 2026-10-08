package reasoning

import "strings"

// IsClaude reports whether model names a Claude model, under any route, region
// or platform prefix.
func IsClaude(model string) bool {
	for _, form := range modelSpellings(model) {
		if _, _, ok := claudeID(claudeName(form)); ok {
			return true
		}
	}
	return false
}

func IsGemini(model string) bool {
	return strings.HasPrefix(baseModelName(model), "gemini-")
}

// ForClaude returns the part of r that may go back to the Claude model target:
// nothing that another model wrote, and no block without a signature. Reasoning
// whose writer is unknown keeps its signed blocks.
func ForClaude(r *ContentReasoning, target string) *ContentReasoning {
	if r.IsEmpty() || writtenByAnother(r, target, IsClaude) {
		return nil
	}
	blocks := r.Sequence()
	signed := make([]Block, 0, len(blocks))
	for _, block := range blocks {
		if block.Redacted != nil || len(block.Signature) > 0 {
			signed = append(signed, block)
		}
	}
	if len(signed) == len(blocks) {
		return r
	}
	return FromBlocks(signed).WrittenBy(r.Model)
}

// ForGemini returns r when its thought signature may go back to the Gemini
// model target, and nil when another model wrote it.
func ForGemini(r *ContentReasoning, target string) *ContentReasoning {
	if writtenByAnother(r, target, IsGemini) {
		return nil
	}
	return r
}

func writtenByAnother(r *ContentReasoning, target string, sameFamily func(string) bool) bool {
	return r != nil && r.Model != "" && r.Model != target && !sameFamily(r.Model)
}
