package reasoning

import "testing"

func TestEveryProviderConstantHasADoorThatPassesIt(t *testing.T) {
	t.Parallel()

	doors := map[Provider]string{
		ProviderUnknown:   "callers with no door of their own",
		ProviderAnthropic: "llms/anthropic",
		ProviderBedrock:   "llms/bedrock",
		ProviderOpenAI:    "llms/openai",
		ProviderGoogleAI:  "llms/googleai",
	}

	if len(doors) != int(providerCount) {
		t.Fatalf("the enum holds %d providers, this test pairs %d of them with a door",
			int(providerCount), len(doors))
	}
	for p := ProviderUnknown; p < providerCount; p++ {
		if doors[p] == "" {
			t.Errorf("provider %d has no door listed", int(p))
		}
	}
}

func TestEveryMistralModelDisablesByOmission(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"mistral-small-latest", "mistral-medium-latest",
		"mistral-large-latest", "magistral-medium-latest",
	} {
		if got := ResolveOff(model, ProviderUnknown); got != OffOmit {
			t.Errorf("ResolveOff(%q) = %v, want OffOmit", model, got)
		}
	}
}
