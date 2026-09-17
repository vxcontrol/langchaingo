package reasoning

import (
	"slices"
	"strings"
)

// IsNovaReasoningModel reports whether the Bedrock model takes Amazon Nova's
// reasoningConfig.
func IsNovaReasoningModel(model string) bool {
	base := baseModelName(model)
	return strings.Contains(base, "amazon.nova-2-lite")
}

// NovaEffort maps a requested effort onto the three levels Nova accepts.
func NovaEffort(effort string) string {
	switch strings.ToLower(effort) {
	case "minimal", "low":
		return "low"
	case "medium":
		return "medium"
	default:
		return "high"
	}
}

func NovaEfforts() []string {
	return []string{"low", "medium", "high"}
}

// NovaClearsInferenceConfigAt reports whether Nova refuses temperature, topP and
// maxTokens alongside the given effort.
func NovaClearsInferenceConfigAt(effort string) bool {
	return NovaEffort(effort) == "high"
}

// IsBedrockAlwaysReasoningModel reports whether the Bedrock family reasons on
// every request and takes no thinking configuration alongside it.
func IsBedrockAlwaysReasoningModel(model string) bool {
	return strings.Contains(baseModelName(model), "deepseek.r1")
}

// IsBedrockNonReasoningModel reports whether the model's Bedrock model card lists
// neither reasoning support nor a request field that controls it.
func IsBedrockNonReasoningModel(model string) bool {
	return slices.Contains(bedrockNonReasoningModels, baseModelName(model))
}

var bedrockNonReasoningModels = []string{"zai.glm-4.7", "zai.glm-4.7-flash", "zai.glm-5"}

// IsGptOssModel reports whether the Bedrock model is one of OpenAI's gpt-oss models.
func IsGptOssModel(model string) bool {
	base := baseModelName(model)
	return strings.Contains(base, "openai.gpt-oss-120b") || strings.Contains(base, "openai.gpt-oss-20b")
}

var gptOssCaps = OpenAIReasoningCaps{Known: true, Efforts: []string{"low", "medium", "high"}}

func GptOssEffort(effort string) string {
	return gptOssCaps.ClampEffort(strings.ToLower(effort))
}

func GptOssEfforts() []string {
	return slices.Clone(gptOssCaps.Efforts)
}

// IsGrokModel reports whether the Bedrock model belongs to the xAI Grok family.
func IsGrokModel(model string) bool {
	return strings.Contains(baseModelName(model), "xai.grok")
}

// GrokEffort maps a requested effort onto what the model accepts. Only the
// generations that kept none can carry it; a request to disable a model that
// always reasons leaves the field off the wire rather than inventing a level.
func GrokEffort(model, effort string) string {
	switch strings.ToLower(effort) {
	case "none":
		if mandatoryThinking(model) {
			return ""
		}
		return "none"
	case "minimal", "low":
		return "low"
	case "medium":
		return "medium"
	case "high":
		return "high"
	case "xhigh", "max":
		return "xhigh"
	default:
		return ""
	}
}
