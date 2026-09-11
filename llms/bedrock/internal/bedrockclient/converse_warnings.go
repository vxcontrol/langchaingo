package bedrockclient

import (
	"encoding/json"
	"strconv"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/vxcontrol/langchaingo/llms"
)

func reportConverseInput(warn *llms.Warnings, input *ConverseInput, built *bedrockruntime.ConverseInput) {
	if built == nil {
		return
	}
	model := input.ModelID
	const (
		omitted   = "the door left it off the converse request"
		different = "the door put a different value on the converse request"
	)

	cfg := built.InferenceConfig
	if input.Temperature != nil {
		sent := (*float32)(nil)
		if cfg != nil {
			sent = cfg.Temperature
		}
		reportConverseFloat(warn, "WithTemperature", model, float32(*input.Temperature), sent)
	}
	if input.TopP != nil {
		sent := (*float32)(nil)
		if cfg != nil {
			sent = cfg.TopP
		}
		reportConverseFloat(warn, "WithTopP", model, float32(*input.TopP), sent)
	}
	if input.MaxTokens != nil && *input.MaxTokens > 0 {
		var sent *int32
		if cfg != nil {
			sent = cfg.MaxTokens
		}
		switch {
		case sent == nil:
			warn.Add(llms.Warning{
				Kind: llms.WarningDrop, Option: "WithMaxTokens", Model: model,
				Asked: strconv.Itoa(*input.MaxTokens), Reason: omitted,
			})
		case int(*sent) != *input.MaxTokens:
			warn.Add(llms.Warning{
				Kind: llms.WarningClamp, Option: "WithMaxTokens", Model: model,
				Asked: strconv.Itoa(*input.MaxTokens), Sent: strconv.FormatInt(int64(*sent), 10),
				Reason: different,
			})
		}
	}
	if len(input.StopSequences) > 0 && (cfg == nil || len(cfg.StopSequences) == 0) {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithStopWords", Model: model,
			Asked: strconv.Itoa(len(input.StopSequences)) + " words", Reason: omitted,
		})
	}
	if input.TopK != nil && *input.TopK != 0 && !converseCarriesTopK(built) {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTopK", Model: model,
			Asked: strconv.Itoa(*input.TopK), Reason: omitted,
		})
	}
	if cfg := input.ReasoningConfig; cfg != nil && cfg.Effort != "" && cfg.Effort != llms.ReasoningNone {
		sent, thinkingSent := converseEffortOnTheWire(built)
		reportEffortClamp(warn, model, string(cfg.Effort), sent, thinkingSent)
	}
	choice, _ := llms.ClassifyToolChoice(input.ToolChoice)
	if choice == llms.ToolChoiceNone &&
		built.ToolConfig != nil && built.ToolConfig.ToolChoice != nil {
		if _, auto := built.ToolConfig.ToolChoice.(*types.ToolChoiceMemberAuto); auto {
			warn.Add(llms.Warning{
				Kind: llms.WarningSubstitute, Option: "WithToolChoice", Model: model,
				Asked: "none", Sent: "auto",
				Reason: "the door builds no none for the converse tool config",
			})
		}
	}
	reportMechanismSwap(warn, model, input.ReasoningConfig, converseMechanismOnTheWire(built))
	if cfg := input.ReasoningConfig; cfg != nil && cfg.HasExplicitTokens() {
		reportThinkingBudget(warn, model, cfg.Tokens, converseThinkingBudget(built))
	}
}

func converseAdditionalFields(built *bedrockruntime.ConverseInput) map[string]any {
	if built.AdditionalModelRequestFields == nil {
		return nil
	}
	raw, err := built.AdditionalModelRequestFields.MarshalSmithyDocument()
	if err != nil {
		return nil
	}
	var fields map[string]any
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil
	}
	return fields
}

// converseEffortOnTheWire reads the three shapes this door writes: the Claude
// output_config, the Nova reasoningConfig and the Grok reasoning object.
func converseEffortOnTheWire(built *bedrockruntime.ConverseInput) (string, bool) {
	fields := converseAdditionalFields(built)
	nested := func(key, effortKey string) (string, bool) {
		block, ok := fields[key].(map[string]any)
		if !ok {
			return "", false
		}
		effort, _ := block[effortKey].(string)
		return effort, true
	}
	for _, shape := range []struct{ key, effortKey string }{
		{"output_config", "effort"},
		{"reasoningConfig", "maxReasoningEffort"},
		{"reasoning", "effort"},
	} {
		if effort, present := nested(shape.key, shape.effortKey); present {
			return effort, true
		}
	}
	_, thinking := fields["thinking"]
	return "", thinking
}

func converseMechanismOnTheWire(built *bedrockruntime.ConverseInput) string {
	fields := converseAdditionalFields(built)
	if thinking, ok := fields["thinking"].(map[string]any); ok {
		sentType, _ := thinking["type"].(string)
		return sentType
	}
	for _, shape := range []struct{ key, effortKey string }{
		{"reasoningConfig", "maxReasoningEffort"},
		{"reasoning", "effort"},
	} {
		block, ok := fields[shape.key].(map[string]any)
		if !ok {
			continue
		}
		if effort, _ := block[shape.effortKey].(string); effort != "" {
			return "effort"
		}
		return ""
	}
	return ""
}

func converseCarriesTopK(built *bedrockruntime.ConverseInput) bool {
	_, carried := converseAdditionalFields(built)["top_k"]
	return carried
}

func converseThinkingBudget(built *bedrockruntime.ConverseInput) int {
	thinking, ok := converseAdditionalFields(built)["thinking"].(map[string]any)
	if !ok {
		return 0
	}
	budget, ok := thinking["budget_tokens"].(float64)
	if !ok {
		return 0
	}
	return int(budget)
}

func reportConverseFloat(warn *llms.Warnings, option, model string, asked float32, sent *float32) {
	render := func(v float32) string { return strconv.FormatFloat(float64(v), 'g', -1, 32) }
	switch {
	case sent == nil:
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: option, Model: model,
			Asked: render(asked), Reason: "the door left it off the converse request",
		})
	case *sent != asked:
		warn.Add(llms.Warning{
			Kind: llms.WarningSubstitute, Option: option, Model: model,
			Asked: render(asked), Sent: render(*sent),
			Reason: "the door put a different value on the converse request",
		})
	}
}
