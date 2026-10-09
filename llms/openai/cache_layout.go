package openai

import (
	"slices"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai/internal/openaiclient"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

var cacheControlRoutes = []string{"anthropic", "bedrock"}

func (o *LLM) placeCacheLayout(req *openaiclient.ChatRequest, layout llms.CacheLayout, model string, warn *llms.Warnings) {
	switch {
	case layout == llms.CacheLayoutNone && o.servedByOpenAI() && reasoning.TakesPromptCacheOptions(model):
		req.PromptCacheOptions = &openaiclient.PromptCacheOptions{Mode: "explicit"}
	case layout != llms.CacheLayoutGrowing || !reasoning.IsClaude(model):
	case o.passesCacheControl(model):
		markClaudeHistory(req, model)
	default:
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithCacheLayout", Model: model, Asked: "growing",
			Reason: "this door places Claude's cache markers only through a LiteLLM route that documents them",
		})
	}
}

func (o *LLM) passesCacheControl(model string) bool {
	route, _, routed := strings.Cut(strings.ToLower(model), "/")
	return reasoning.IsClaude(model) && !reasoning.PublicProviderHost(o.host) && o.host != anthropicAPIHost &&
		(!routed || slices.Contains(cacheControlRoutes, route))
}

func markClaudeHistory(req *openaiclient.ChatRequest, model string) {
	route, _, _ := strings.Cut(strings.ToLower(model), "/")
	control := &openaiclient.CacheControl{Type: "ephemeral", TTL: "1h"}
	if route == "bedrock" && reasoning.BedrockCachesFiveMinutesOnly(model) {
		control.TTL = ""
	}
	if !markLastText(req.Messages, RoleSystem, control) && route != "bedrock" && len(req.Tools) > 0 {
		req.Tools[len(req.Tools)-1].Function.CacheControl = control
	}
	markLastText(req.Messages, RoleUser, control)
}

func markLastText(messages []*ChatMessage, role string, control *openaiclient.CacheControl) bool {
	for _, msg := range slices.Backward(messages) {
		if msg.Role != role {
			continue
		}
		for j, part := range slices.Backward(msg.MultiContent) {
			if text, ok := part.(llms.TextContent); ok {
				msg.MultiContent[j] = openaiclient.CachedText{TextContent: text, CacheControl: control}
				return true
			}
		}
	}
	return false
}
