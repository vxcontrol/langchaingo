package bedrock

import (
	"strconv"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock/internal/bedrockclient"
)

func legacyCarriesPenalties(model string) bool {
	return bedrockclient.GetProvider(model) == "ai21" && !bedrockclient.IsAi21Jamba(model)
}

func legacyCarriesCandidateCount(model string) bool {
	switch bedrockclient.GetProvider(model) {
	case "ai21":
		return true
	case "cohere":
		return !bedrockclient.IsCohereCommandR(model)
	}
	return false
}

func unreadBedrockOptions(model string, converse bool, opts llms.CallOptions) []llms.Warning {
	const unread = "the bedrock request for this model has no field for it"

	carried := []string{"WithTopK"}
	if !converse {
		if legacyCarriesPenalties(model) {
			carried = append(carried,
				"WithRepetitionPenalty", "WithFrequencyPenalty", "WithPresencePenalty")
		}
		if legacyCarriesCandidateCount(model) {
			carried = append(carried, "WithCandidateCount")
		}
	}

	var warn llms.Warnings
	warn.AddUnreadExtraBody(model, opts, extraBodyUnread)
	warn.AddUnreadOptions(model, opts, unread, carried...)
	return warn.List()
}

const extraBodyUnread = "the door builds its request through a vendor SDK and has nowhere to merge them"

func toolsWithAFunction(warn *llms.Warnings, model string, tools []llms.Tool) []llms.Tool {
	sent := make([]llms.Tool, 0, len(tools))
	for _, tool := range tools {
		if tool.Function != nil {
			sent = append(sent, tool)
		}
	}
	if dropped := len(tools) - len(sent); dropped > 0 {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTools", Model: model,
			Asked:  strconv.Itoa(dropped) + " tools",
			Reason: "a tool without a function definition has nothing to send",
		})
		return sent
	}
	return tools
}
