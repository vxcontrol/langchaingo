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
	if input.TopK != nil && !converseCarriesTopK(built) {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTopK", Model: model,
			Asked: strconv.Itoa(*input.TopK), Reason: omitted,
		})
	}
	if cfg := input.ReasoningConfig; cfg != nil && cfg.Effort != "" && cfg.Effort != llms.ReasoningNone {
		fields := converseAdditionalFields(built)
		sent := ""
		if oc, ok := fields["output_config"].(map[string]any); ok {
			sent, _ = oc["effort"].(string)
		}
		_, thinkingSent := fields["thinking"]
		reportEffortClamp(warn, model, string(cfg.Effort), sent, thinkingSent)
	}
	if kind, _ := llms.ClassifyToolChoice(input.ToolChoice); kind == llms.ToolChoiceNone &&
		built.ToolConfig != nil && built.ToolConfig.ToolChoice != nil {
		if _, auto := built.ToolConfig.ToolChoice.(*types.ToolChoiceMemberAuto); auto {
			warn.Add(llms.Warning{
				Kind: llms.WarningSubstitute, Option: "WithToolChoice", Model: model,
				Asked: "none", Sent: "auto",
				Reason: "the door builds no none for the converse tool config",
			})
		}
	}
	if cfg := input.ReasoningConfig; cfg != nil && cfg.HasExplicitTokens() {
		reportThinkingBudget(warn, model, cfg.Tokens, converseThinkingBudget(built))
	}
	if len(input.Tools) > 0 && built.ToolConfig == nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTools", Model: model,
			Asked: strconv.Itoa(len(input.Tools)) + " tools", Reason: omitted,
		})
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
