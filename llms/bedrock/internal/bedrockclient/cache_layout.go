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
	points := map[int]types.CacheTTL{}
	latestHour := -1
	for _, at := range []int{converseMovingPoint(messages, blocks), converseTurnStart(messages, blocks)} {
		if block, ok := markableBlock(messages, blocks, at); ok {
			points[block] = hour
			latestHour = max(latestHour, block)
		}
	}
	if tail, ok := markableBlock(messages, blocks, len(blocks)-1); ok && tail > latestHour {
		points[tail] = types.CacheTTLFiveMinutes
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

func converseMovingPoint(messages []types.Message, blocks []converseBlock) int {
	moving := -1
	for b, at := range blocks {
		last := b == len(blocks)-1
		if !last && (messages[at.message].Role != types.ConversationRoleUser || blocks[b+1].message == at.message) {
			continue
		}
		line := b / converseGrowingCacheStep * converseGrowingCacheStep
		if moving >= 0 {
			line = min(line, moving+converseGrowingCacheStep)
		}
		moving = max(moving, line)
	}
	return moving
}

func converseTurnStart(messages []types.Message, blocks []converseBlock) int {
	for b := len(blocks) - 1; b >= 0; b-- {
		msg := messages[blocks[b].message]
		if msg.Role != types.ConversationRoleUser {
			continue
		}
		if _, isResult := msg.Content[blocks[b].block].(*types.ContentBlockMemberToolResult); !isResult {
			return b
		}
	}
	return -1
}

func markableBlock(messages []types.Message, blocks []converseBlock, at int) (int, bool) {
	if at < 0 {
		return 0, false
	}
	message := blocks[at].message
	for b := at; b < len(blocks) && blocks[b].message == message; b++ {
		if isMarkable(messages, blocks[b]) {
			return b, true
		}
	}
	for b := at - 1; b >= 0 && blocks[b].message == message; b-- {
		if isMarkable(messages, blocks[b]) {
			return b, true
		}
	}
	return 0, false
}

func isMarkable(messages []types.Message, at converseBlock) bool {
	_, reasoning := messages[at.message].Content[at.block].(*types.ContentBlockMemberReasoningContent)
	return !reasoning
}

func dropCachePoints(system *[]types.SystemContentBlock, messages []types.Message) int {
	before := len(*system)
	*system = slices.DeleteFunc(*system, func(block types.SystemContentBlock) bool {
		_, point := block.(*types.SystemContentBlockMemberCachePoint)
		return point
	})
	dropped := before - len(*system)
	for i := range messages {
		before := len(messages[i].Content)
		messages[i].Content = slices.DeleteFunc(messages[i].Content, func(block types.ContentBlock) bool {
			_, point := block.(*types.ContentBlockMemberCachePoint)
			return point
		})
		dropped += before - len(messages[i].Content)
	}
	return dropped
}
