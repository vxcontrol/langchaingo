package anthropic

import (
	"github.com/vxcontrol/langchaingo/llms/anthropic/internal/anthropicclient"
)

const growingCacheStep = 15

type cachePosition struct {
	message, block int
}

func placeGrowingCacheMarkers(tools []anthropicclient.Tool, systemPrompt *any, messages []anthropicclient.ChatMessage) {
	hour := &anthropicclient.CacheControl{Type: "ephemeral", TTL: "1h"}
	if !markSystem(systemPrompt, hour) && len(tools) > 0 {
		tools[len(tools)-1].CacheControl = hour
	}

	positions := cachePositions(messages)
	if len(positions) == 0 {
		return
	}
	last := len(positions) - 1
	hourLong := map[int]bool{last / growingCacheStep * growingCacheStep: true}
	if start, ok := turnStart(messages, positions); ok {
		hourLong[start] = true
	}
	for position := range hourLong {
		markAt(messages, positions[position], hour)
	}
	if !hourLong[last] {
		markAt(messages, positions[last], &anthropicclient.CacheControl{Type: "ephemeral", TTL: "5m"})
	}
}

func markSystem(systemPrompt *any, cacheControl *anthropicclient.CacheControl) bool {
	switch system := (*systemPrompt).(type) {
	case string:
		if system == "" {
			return false
		}
		*systemPrompt = []anthropicclient.Content{
			&anthropicclient.TextContent{Type: "text", Text: system, CacheControl: cacheControl},
		}
		return true
	case []anthropicclient.Content:
		for i := len(system) - 1; i >= 0; i-- {
			if markBlock(system[i], cacheControl) {
				return true
			}
		}
	}
	return false
}

func cachePositions(messages []anthropicclient.ChatMessage) []cachePosition {
	var positions []cachePosition
	for i, msg := range messages {
		for j := range msg.Content {
			positions = append(positions, cachePosition{message: i, block: j})
		}
	}
	return positions
}

func turnStart(messages []anthropicclient.ChatMessage, positions []cachePosition) (int, bool) {
	for p := len(positions) - 1; p >= 0; p-- {
		msg := messages[positions[p].message]
		if msg.Role != RoleUser {
			continue
		}
		for _, block := range msg.Content {
			if block.GetType() != "tool_result" {
				return lastPositionOf(positions, positions[p].message), true
			}
		}
	}
	return 0, false
}

func lastPositionOf(positions []cachePosition, message int) int {
	last := 0
	for p, position := range positions {
		if position.message == message {
			last = p
		}
	}
	return last
}

func markAt(messages []anthropicclient.ChatMessage, position cachePosition, cacheControl *anthropicclient.CacheControl) {
	content := messages[position.message].Content
	for j := position.block; j < len(content); j++ {
		if markBlock(content[j], cacheControl) {
			return
		}
	}
	for j := position.block - 1; j >= 0; j-- {
		if markBlock(content[j], cacheControl) {
			return
		}
	}
}

func markBlock(block anthropicclient.Content, cacheControl *anthropicclient.CacheControl) bool {
	field := cacheControlOf(block)
	if field != nil {
		*field = cacheControl
	}
	return field != nil
}

func cacheControlOf(block anthropicclient.Content) **anthropicclient.CacheControl {
	switch c := block.(type) {
	case *anthropicclient.TextContent:
		return &c.CacheControl
	case *anthropicclient.ToolResultContent:
		return &c.CacheControl
	case *anthropicclient.ImageContent:
		return &c.CacheControl
	case *anthropicclient.ToolUseContent:
		return &c.CacheControl
	}
	return nil
}

func dropCacheMarkers(systemPrompt any, messages []anthropicclient.ChatMessage) int {
	system, _ := systemPrompt.([]anthropicclient.Content)
	dropped := dropMarkers(system)
	for _, msg := range messages {
		dropped += dropMarkers(msg.Content)
	}
	return dropped
}

func dropMarkers(blocks []anthropicclient.Content) int {
	dropped := 0
	for _, block := range blocks {
		if field := cacheControlOf(block); field != nil && *field != nil {
			*field = nil
			dropped++
		}
	}
	return dropped
}
