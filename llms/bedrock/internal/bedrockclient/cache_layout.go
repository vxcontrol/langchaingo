package bedrockclient

import (
	"slices"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const converseGrowingCacheStep = 12

type converseBlock struct {
	message, block int
}

func placeGrowingCachePoints(system *[]types.SystemContentBlock, messages []types.Message, oneHour bool) {
	hour := types.CacheTTLFiveMinutes
	if oneHour {
		hour = types.CacheTTLOneHour
	}
	if len(*system) > 0 {
		*system = append(*system, &types.SystemContentBlockMemberCachePoint{
			Value: types.CachePointBlock{Type: types.CachePointTypeDefault, Ttl: hour},
		})
	}

	var blocks []converseBlock
	for i, msg := range messages {
		for j := range msg.Content {
			blocks = append(blocks, converseBlock{message: i, block: j})
		}
	}
	if len(blocks) == 0 {
		return
	}
	last := len(blocks) - 1
	points := map[int]types.CacheTTL{last / converseGrowingCacheStep * converseGrowingCacheStep: hour}
	if start, ok := converseTurnStart(messages, blocks); ok {
		points[start] = hour
	}
	if _, ok := points[last]; !ok {
		points[last] = types.CacheTTLFiveMinutes
	}

	at := make([]int, 0, len(points))
	for block := range points {
		at = append(at, block)
	}
	slices.Sort(at)
	for _, block := range slices.Backward(at) {
		target := blocks[block]
		content := messages[target.message].Content
		point := &types.ContentBlockMemberCachePoint{Value: types.CachePointBlock{Type: types.CachePointTypeDefault, Ttl: points[block]}}
		messages[target.message].Content = slices.Insert(content, target.block+1, types.ContentBlock(point))
	}
}

func converseTurnStart(messages []types.Message, blocks []converseBlock) (int, bool) {
	for b := len(blocks) - 1; b >= 0; b-- {
		msg := messages[blocks[b].message]
		if msg.Role != types.ConversationRoleUser {
			continue
		}
		if _, isResult := msg.Content[blocks[b].block].(*types.ContentBlockMemberToolResult); !isResult {
			return b, true
		}
	}
	return 0, false
}
