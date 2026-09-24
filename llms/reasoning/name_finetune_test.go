package reasoning

import "testing"

func TestAFineTunedNameIsClassifiedAsItsBaseModel(t *testing.T) {
	const (
		base      = "gpt-4o-2024-08-06"
		fineTuned = "ft:gpt-4o-2024-08-06:acme::9xYzAbCd"
	)

	if RejectsTopK(base) != RejectsTopK(fineTuned) {
		t.Errorf("RejectsTopK: base %v, fine-tuned %v", RejectsTopK(base), RejectsTopK(fineTuned))
	}
	if RejectsRepetitionPenalty(base) != RejectsRepetitionPenalty(fineTuned) {
		t.Error("RejectsRepetitionPenalty disagrees between a base model and its fine-tune")
	}
	if !RejectsTopK(fineTuned) {
		t.Error("top_k must stay off the wire for a fine-tune of an OpenAI model")
	}
}

func TestTheWrapperIsStrippedWithAndWithoutTheTrailingFields(t *testing.T) {
	for _, model := range []string{
		"ft:gpt-4o-2024-08-06:acme::9xYzAbCd",
		"ft:gpt-4o-2024-08-06",
		"ft:o3-mini:acme::1",
	} {
		if !RejectsTopK(model) {
			t.Errorf("%s: top_k must stay off the wire, as it does for the base model", model)
		}
	}
	if RejectsTopK("ft:") {
		t.Error("an empty wrapper names no model and must not classify as one")
	}
	if RejectsTopK("ft:claude-opus-5:acme::1") {
		t.Error("a fine-tune of another vendor's model is not an OpenAI name")
	}
}
