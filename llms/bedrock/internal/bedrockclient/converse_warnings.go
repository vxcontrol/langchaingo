package bedrockclient

import (
	"encoding/json"
	"strconv"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func reportConverseInput(warn *llms.Warnings, input *ConverseInput, built *bedrockruntime.ConverseInput) {
	if built == nil {
		return
	}
	model := input.ModelID
	sentThinking := converseMechanismOnTheWire(built)
	reportClaudeOffFloor(warn, model, sentThinking)
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
		if sent != nil && !thinkingSetsTheTemperature(sentThinking) {
			reportTemperatureClamp(warn, model, *input.Temperature)
			reportConverseRangeClamp(warn, "WithTemperature", model,
				*input.Temperature, clampTemperature(model, *input.Temperature), *sent)
		} else {
			reportConverseFloat(warn, "WithTemperature", model, float32(*input.Temperature), sent)
		}
	}
	if input.TopP != nil {
		sent := (*float32)(nil)
		if cfg != nil {
			sent = cfg.TopP
		}
		if sent != nil {
			nova := reasoning.NovaClampTopP(model, *input.TopP)
			reportNovaClamp(warn, "WithTopP", "topP", model, *input.TopP, nova)
			reportConverseRangeClamp(warn, "WithTopP", model, *input.TopP, nova, *sent)
		} else {
			reportConverseFloat(warn, "WithTopP", model, float32(*input.TopP), sent)
		}
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
	if asked := len(nonEmptyStops(input.StopSequences)); asked > 0 {
		sent := 0
		if cfg != nil {
			sent = len(cfg.StopSequences)
		}
		switch {
		case sent == 0:
			warn.Add(llms.Warning{
				Kind: llms.WarningDrop, Option: "WithStopWords", Model: model,
				Asked: strconv.Itoa(len(input.StopSequences)) + " words", Reason: omitted,
			})
		case sent < asked:
			warn.Add(llms.Warning{
				Kind: llms.WarningClamp, Option: "WithStopWords", Model: model,
				Asked: strconv.Itoa(len(input.StopSequences)) + " words", Sent: strconv.Itoa(sent) + " words",
				Reason: "the Converse API takes at most " + strconv.Itoa(converseMaxStopSequences) + " stop sequences",
			})
		}
	}
	if input.TopK != nil && *input.TopK != 0 {
		if sent, carried := converseTopKOnTheWire(built); carried {
			reportTopKClamp(warn, model, *input.TopK, sent)
		} else {
			warn.Add(llms.Warning{
				Kind: llms.WarningDrop, Option: "WithTopK", Model: model,
				Asked: strconv.Itoa(*input.TopK), Reason: omitted,
			})
		}
	}
	if converseDropsThinkingAtNoNamedDepth(model, input.ReasoningConfig) {
		reportThinkingUnsupported(warn, model, input.ReasoningConfig)
	}
	if cfg := input.ReasoningConfig; cfg != nil && cfg.Effort != "" && cfg.Effort != llms.ReasoningNone {
		sent, thinkingSent := converseEffortOnTheWire(built)
		reportEffortClamp(warn, model, string(cfg.Effort), sent, thinkingSent,
			cfg, converseThinkingBudget(built))
	}
	if built.ToolConfig == nil {
		warn.AddToolChoiceWithoutTools(model, input.ToolChoice)
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
	if effort := converseNovaDelegatedEffort(input, built); effort != "" {
		reportDelegatedEffort(warn, model, effort)
	} else {
		reportMechanismSwap(warn, model, input.ReasoningConfig, converseMechanismOnTheWire(built))
	}
	if cfg := input.ReasoningConfig; cfg != nil && cfg.HasExplicitTokens() {
		reportThinkingBudget(warn, model, cfg.Tokens, converseThinkingBudget(built))
	}
}

func converseNovaDelegatedEffort(input *ConverseInput, built *bedrockruntime.ConverseInput) string {
	if !input.ReasoningConfig.DelegatesDepth() {
		return ""
	}
	config, _ := converseAdditionalFields(built)["reasoningConfig"].(map[string]any)
	effort, _ := config["maxReasoningEffort"].(string)
	return effort
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

// converseEffortOnTheWire reads the effort from every shape this door writes.
func converseEffortOnTheWire(built *bedrockruntime.ConverseInput) (string, bool) {
	fields := converseAdditionalFields(built)
	if effort, ok := fields["reasoning_effort"].(string); ok {
		return effort, true
	}
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
	if effort, _ := fields["reasoning_effort"].(string); effort != "" {
		return "effort"
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

func converseDropsThinkingAtNoNamedDepth(model string, cfg *llms.ReasoningConfig) bool {
	if cfg.ResolveMode() != llms.ReasoningOn || cfg.Effort != llms.ReasoningNone || cfg.HasExplicitTokens() {
		return false
	}
	if reasoning.IsReasoningModel(model) && !reasoning.IsBedrockNonReasoningModel(model) {
		return false
	}
	return reasoning.ResolveMechanism(model, cfg.Adaptive, isAnthropicModelID(model), false) == reasoning.MechanismNone
}

func converseTopKOnTheWire(built *bedrockruntime.ConverseInput) (int, bool) {
	fields := converseAdditionalFields(built)
	topK, carried := fields["top_k"]
	if !carried {
		novaConfig, _ := fields["inferenceConfig"].(map[string]any)
		topK, carried = novaConfig["topK"]
	}
	value, _ := topK.(float64)
	return int(value), carried
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

func reportConverseRangeClamp(warn *llms.Warnings, option, model string, asked, family float64, sent float32) {
	if float32(family) == sent {
		return
	}
	warn.Add(llms.Warning{
		Kind: llms.WarningClamp, Option: option, Model: model,
		Asked:  strconv.FormatFloat(asked, 'g', -1, 64),
		Sent:   strconv.FormatFloat(float64(sent), 'g', -1, 32),
		Reason: "the Converse API takes a value from 0 to 1",
	})
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
