package huggingface

import (
	"strconv"

	"github.com/vxcontrol/langchaingo/llms"
)

func reportHuggingFaceOptions(
	warn *llms.Warnings, model string, opts *llms.CallOptions, messages []llms.MessageContent,
) {
	const unread = "the door builds no field for it"

	drop := func(option, asked string) {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: option, Model: model,
			Asked: asked, Reason: unread,
		})
	}
	dropFloat := func(option string, v *float64) {
		if v != nil && *v != 0 {
			drop(option, strconv.FormatFloat(*v, 'g', -1, 64))
		}
	}
	dropInt := func(option string, v *int) {
		if v != nil && *v != 0 {
			drop(option, strconv.Itoa(*v))
		}
	}

	dropInt("WithMaxTokens", opts.MaxTokens)
	dropInt("WithTopK", opts.TopK)
	dropInt("WithMinLength", opts.MinLength)
	dropCount := func(option string, v *int) {
		if v != nil && *v != 1 {
			drop(option, strconv.Itoa(*v))
		}
	}
	dropCount("WithN", opts.N)
	dropCount("WithCandidateCount", opts.CandidateCount)
	dropInt("WithTopLogProbs", opts.TopLogProbs)
	dropFloat("WithMinP", opts.MinP)
	dropFloat("WithRepetitionPenalty", opts.RepetitionPenalty)
	dropFloat("WithFrequencyPenalty", opts.FrequencyPenalty)
	dropFloat("WithPresencePenalty", opts.PresencePenalty)

	reportHuggingFaceShapedOptions(opts, drop)
	if cfg := opts.Reasoning; cfg != nil && cfg.HasExplicitTokens() {
		drop("WithReasoning", strconv.Itoa(cfg.Tokens)+" tokens")
	}
	reportHuggingFaceMessages(warn, model, messages)
}

func reportHuggingFaceShapedOptions(opts *llms.CallOptions, drop func(option, asked string)) {
	if opts.LogProbs != nil && *opts.LogProbs {
		drop("WithLogProbs", "true")
	}
	if len(opts.StopWords) > 0 {
		drop("WithStopWords", strconv.Itoa(len(opts.StopWords))+" words")
	}
	if len(opts.Tools) > 0 {
		drop("WithTools", strconv.Itoa(len(opts.Tools))+" tools")
	}
	if len(opts.Functions) > 0 {
		drop("WithFunctions", strconv.Itoa(len(opts.Functions))+" functions")
	}
	if kind, name := llms.ClassifyToolChoice(opts.ToolChoice); kind != llms.ToolChoiceUnset &&
		kind != llms.ToolChoiceAuto {
		asked := name
		if asked == "" {
			asked = kind.String()
		}
		drop("WithToolChoice", asked)
	}
	if opts.StreamingFunc != nil {
		drop("WithStreamingFunc", "a callback")
	}
	if opts.StructuredOutput != nil {
		drop("WithStructuredOutput", opts.StructuredOutput.Name)
	}
	if opts.JSONMode {
		drop("WithJSONMode", "true")
	}
	if opts.ResponseMIMEType != nil {
		drop("WithResponseMIMEType", *opts.ResponseMIMEType)
	}
}

func reportHuggingFaceMessages(warn *llms.Warnings, model string, messages []llms.MessageContent) {
	carried := 0
	if len(messages) > 0 && len(messages[0].Parts) > 0 {
		carried = 1
	}
	total := 0
	for _, m := range messages {
		total += len(m.Parts)
	}
	if total > carried {
		warn.Add(llms.Warning{
			Kind: llms.WarningClamp, Option: "messages", Model: model,
			Asked: strconv.Itoa(total) + " parts", Sent: strconv.Itoa(carried) + " part",
			Reason: "the door sends the first part of the first message as the whole prompt",
		})
	}
}
