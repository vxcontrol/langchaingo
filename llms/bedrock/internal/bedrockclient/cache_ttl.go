package bedrockclient

import (
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func fitConverseCacheTTLs(warn *llms.Warnings, modelID string, built *bedrockruntime.ConverseInput) {
	if !reasoning.BedrockCachesFiveMinutesOnly(modelID) {
		return
	}
	oneHour := false
	fit := func(point *types.CachePointBlock) {
		oneHour = oneHour || point.Ttl == types.CacheTTLOneHour
		point.Ttl = ""
	}
	for _, block := range built.System {
		if point, ok := block.(*types.SystemContentBlockMemberCachePoint); ok {
			fit(&point.Value)
		}
	}
	for _, message := range built.Messages {
		for _, block := range message.Content {
			if point, ok := block.(*types.ContentBlockMemberCachePoint); ok {
				fit(&point.Value)
			}
		}
	}
	if oneHour {
		reportOneHourCacheTTL(warn, modelID)
	}
}

func fitAnthropicCacheTTLs(warn *llms.Warnings, modelID string, messages []*anthropicTextGenerationInputMessage) {
	if !reasoning.BedrockCachesFiveMinutesOnly(modelID) {
		return
	}
	oneHour := false
	for _, message := range messages {
		for i := range message.Content {
			if mark := message.Content[i].CacheControl; mark != nil {
				oneHour = oneHour || mark.TTL == "1h"
				mark.TTL = ""
			}
		}
	}
	if oneHour {
		reportOneHourCacheTTL(warn, modelID)
	}
}

func reportOneHourCacheTTL(warn *llms.Warnings, modelID string) {
	warn.Add(llms.Warning{
		Kind: llms.WarningSubstitute, Option: "WithCacheControl", Model: modelID,
		Asked: "1h", Sent: "5m", Reason: "AWS gives this model the 5-minute cache TTL only",
	})
}
