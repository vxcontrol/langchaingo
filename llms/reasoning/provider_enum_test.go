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
		ProviderOllama:    "llms/ollama",
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

func TestOllamaDisablesByBooleanEvenForNamesOtherVendorsServeAsMandatory(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"deepseek-r1:7b", "qwen3:8b", "magistral-small:24b"} {
		if got := ResolveOff(model, ProviderOllama); got != OffDisableThinkBool {
			t.Errorf("ResolveOff(%q, ProviderOllama) = %v, want OffDisableThinkBool", model, got)
		}
		if same := ResolveOff(model, ProviderUnknown); same == OffDisableThinkBool {
			t.Errorf("ResolveOff(%q, ProviderUnknown) already answers for the ollama door", model)
		}
	}
}

func TestOllamaRefusesToDisableTheFamilyThatIgnoresBooleans(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-oss:120b", "gpt-oss:20b", "library/gpt-oss:120b"} {
		if got := ResolveOff(model, ProviderOllama); got != OffUnsupported {
			t.Errorf("ResolveOff(%q, ProviderOllama) = %v, want OffUnsupported", model, got)
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

func TestOnlyOpenAIsOwnCatalogueRefusesTopK(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-4o", "gpt-5.4-nano", "o3", "o4-mini", "chatgpt-4o-latest"} {
		if !RejectsTopK(model) {
			t.Errorf("%q is OpenAI's own name; its endpoint fails the call on top_k", model)
		}
	}
	for _, model := range []string{"gpt-oss:20b", "gpt-oss-120b", "zai/glm-4.5-air", "qwen3-max", "grok-4"} {
		if RejectsTopK(model) {
			t.Errorf("%q is not an OpenAI name and must not inherit its refusal", model)
		}
	}
}
