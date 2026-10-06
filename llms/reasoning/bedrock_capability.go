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

const NovaDelegatedEffort = "medium"

func NovaEfforts() []string {
	return []string{"low", "medium", "high"}
}

func NovaClearsInferenceConfigAt(effort string) bool {
	return NovaEffort(effort) == "high"
}

const NovaMinTemperature = 0.00001

func NovaClampTemperature(model string, temperature float64) float64 {
	m := bedrockModelWithoutRegion(model)
	for _, family := range []string{"amazon.nova-micro", "amazon.nova-lite", "amazon.nova-pro", "amazon.nova-premier", "amazon.nova-2-"} {
		if strings.HasPrefix(m, family) {
			return min(max(temperature, NovaMinTemperature), 1)
		}
	}
	return temperature
}

// IsBedrockAlwaysReasoningModel reports whether the Bedrock family reasons on
// every request and takes no thinking configuration alongside it.
func IsBedrockAlwaysReasoningModel(model string) bool {
	return strings.Contains(baseModelName(model), "deepseek.r1")
}

// IsBedrockNonReasoningModel reports whether the model is in
// bedrockNonReasoningModels: Bedrock models whose AWS model card lists neither
// reasoning support nor a request field that controls it.
func IsBedrockNonReasoningModel(model string) bool {
	return slices.Contains(bedrockNonReasoningModels, bedrockModelWithoutRegion(model))
}

var bedrockNonReasoningModels = []string{"zai.glm-4.7", "zai.glm-4.7-flash", "zai.glm-5"}

// BedrockSupportsStructuredOutput reports whether the AWS model card of a
// non-Claude Bedrock model lists structured outputs for the bedrock-runtime
// endpoint. Claude follows ClaudeSupportsStructuredOutputOnBedrock.
func BedrockSupportsStructuredOutput(model string) bool {
	return slices.Contains(bedrockStructuredOutputModels, bedrockModelWithoutRegion(model))
}

var bedrockStructuredOutputModels = []string{
	"deepseek.v3-v1:0", "deepseek.v3.1", "deepseek.v3.2",
	"google.gemma-3-12b-it", "google.gemma-3-27b-it",
	"minimax.minimax-m2", "minimax.minimax-m2.1", "minimax.minimax-m2.5",
	"mistral.devstral-2-123b", "mistral.magistral-small-2509", "mistral.ministral-3-14b-instruct",
	"mistral.ministral-3-8b-instruct", "mistral.ministral-3-3b-instruct", "mistral.mistral-large-3-675b-instruct",
	"mistral.voxtral-mini-3b-2507", "mistral.voxtral-small-24b-2507",
	"moonshot.kimi-k2-thinking", "moonshotai.kimi-k2-thinking", "moonshotai.kimi-k2.5", "moonshotai.kimi-k3",
	"nvidia.nemotron-nano-12b-v2", "nvidia.nemotron-nano-3-30b", "nvidia.nemotron-nano-9b-v2",
	"nvidia.nemotron-super-3-120b",
	"openai.gpt-5.6-luna", "openai.gpt-5.6-sol", "openai.gpt-5.6-terra",
	"openai.gpt-6-astra", "openai.gpt-6-luna", "openai.gpt-6-sol", "openai.gpt-6.1-sol",
	"openai.gpt-oss-120b", "openai.gpt-oss-120b-1:0", "openai.gpt-oss-20b",
	"openai.gpt-oss-20b-1:0", "openai.gpt-oss-safeguard-120b", "openai.gpt-oss-safeguard-20b",
	"qwen.qwen3-235b-a22b-2507", "qwen.qwen3-235b-a22b-2507-v1:0", "qwen.qwen3-32b", "qwen.qwen3-32b-v1:0",
	"qwen.qwen3-coder-30b-a3b-instruct", "qwen.qwen3-coder-30b-a3b-v1:0", "qwen.qwen3-coder-480b-a35b-instruct",
	"qwen.qwen3-coder-480b-a35b-v1:0", "qwen.qwen3-coder-next", "qwen.qwen3-next-80b-a3b",
	"qwen.qwen3-next-80b-a3b-instruct",
	"writer.palmyra-vision-7b",
	"xai.grok-4.7",
	"zai.glm-4.7", "zai.glm-4.7-flash", "zai.glm-5",
}

func BedrockStructuredOutputNeedsStrict(model string) bool {
	m := bedrockModelWithoutRegion(model)
	return strings.HasPrefix(m, "openai.gpt-5.6-") || strings.HasPrefix(m, "openai.gpt-6")
}

func BedrockSchemaOnNonStreamingOnly(model string) bool {
	return BedrockStructuredOutputNeedsStrict(model) &&
		!strings.HasPrefix(bedrockModelWithoutRegion(model), "openai.gpt-6.1-sol")
}

var bedrockRegionPrefixes = []string{"us-gov.", "apac.", "global.", "us.", "eu.", "au.", "jp.", "in.", "ca."}

func bedrockModelWithoutRegion(model string) string {
	base := baseModelName(model)
	for _, prefix := range bedrockRegionPrefixes {
		if trimmed, found := strings.CutPrefix(base, prefix); found {
			base = trimmed
			break
		}
	}
	if _, bare := splitPlatformPrefix(base); bare != base {
		if documented, inherited := inheritLine(bare); inherited {
			return strings.TrimSuffix(base, bare) + documented
		}
	}
	return base
}

// IsGptOssModel reports whether the model is one of OpenAI's gpt-oss models,
// in any host's spelling.
func IsGptOssModel(model string) bool {
	for _, form := range modelSpellings(model) {
		if strings.HasPrefix(form, "gpt-oss") {
			return true
		}
	}
	return false
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

func BedrockRejectsReasoningReplay(model string) bool {
	return strings.HasPrefix(bedrockModelWithoutRegion(model), "moonshotai.kimi-k3")
}
