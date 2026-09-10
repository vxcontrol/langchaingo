package bedrockclient

import (
	"strconv"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

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
	if len(input.Tools) > 0 && built.ToolConfig == nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTools", Model: model,
			Asked: strconv.Itoa(len(input.Tools)) + " tools", Reason: omitted,
		})
	}
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
