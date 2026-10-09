package bedrockclient

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/require"
)

func growingHistory(steps []string) []types.Message {
	history := []types.Message{{Role: types.ConversationRoleUser, Content: []types.ContentBlock{
		&types.ContentBlockMemberText{Value: "scan the host"},
	}}}
	for i, shape := range steps {
		var answer, results []types.ContentBlock
		for j, kind := range shape {
			switch kind {
			case 'R':
				answer = append(answer, &types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberReasoningText{Value: types.ReasoningTextBlock{Text: aws.String("plan")}},
				})
			case 'T':
				answer = append(answer, &types.ContentBlockMemberText{Value: "checking"})
			case 'U':
				id := fmt.Sprintf("call_%d_%d", i, j)
				answer = append(answer, &types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{ToolUseId: aws.String(id)}})
				results = append(results, &types.ContentBlockMemberToolResult{Value: types.ToolResultBlock{ToolUseId: aws.String(id)}})
			}
		}
		history = append(history,
			types.Message{Role: types.ConversationRoleAssistant, Content: answer},
			types.Message{Role: types.ConversationRoleUser, Content: results})
	}
	return history
}

func movingPoints(t *testing.T, steps []string) []int {
	t.Helper()

	history := growingHistory(steps)
	var points []int
	for end := 1; end <= len(history); end += 2 {
		messages := make([]types.Message, end)
		for i, msg := range history[:end] {
			messages[i] = types.Message{Role: msg.Role, Content: slices.Clone(msg.Content)}
		}
		var system []types.SystemContentBlock
		placeGrowingCachePoints(&system, messages, true)

		furthest, block := -1, -1
		var previous types.ContentBlock
		for _, msg := range messages {
			for _, content := range msg.Content {
				point, ok := content.(*types.ContentBlockMemberCachePoint)
				if !ok {
					block++
					previous = content
					continue
				}
				_, reasoning := previous.(*types.ContentBlockMemberReasoningContent)
				require.False(t, reasoning, "%v: request %d: a point after reasoning", steps, end/2)
				if point.Value.Ttl == types.CacheTTLOneHour {
					furthest = max(furthest, block)
				}
			}
		}
		points = append(points, furthest)
	}
	return points
}

func TestEachRequestsHourLongWriteIsInReachOfTheNext(t *testing.T) {
	t.Parallel()

	shapes := []string{"U", "RU", "TU", "RTU", "RUU", "UUR", "RRU", "RRRU", "RRUUU", "RTUUU", "RTUUUUUU",
		"UUUUUUUUR", "RUUUUUUUUU", "RUUUUUUUUUU", "RTUUUUUUUUUUU"}
	histories := [][]string{
		{"UUUUUUUUR", "RUUUUUUUUUU", "RUUUUUUUUUU", "RUUUUUUUUU", "RUUUUUUUUUU", "RUUUUUUUUU", "RRRU"},
	}
	rng := rand.New(rand.NewSource(1)) //nolint:gosec
	for range 3000 {
		steps := make([]string, 2+rng.Intn(30))
		for i := range steps {
			steps[i] = shapes[rng.Intn(len(shapes))]
		}
		histories = append(histories, steps)
	}

	for _, steps := range histories {
		points := movingPoints(t, steps)
		for i := 1; i < len(points); i++ {
			require.GreaterOrEqual(t, points[i], points[i-1], "%v: request %d", steps, i)
			require.LessOrEqual(t, points[i]-points[i-1], 18, "%v: request %d is out of reach of the write at %d", steps, i, points[i-1])
		}
	}
}

func TestTheHourLongWriteKeepsUpWithStepsOfEighteenBlocks(t *testing.T) {
	t.Parallel()

	for _, shape := range []string{"U", "RTU", "RRUUU", "RTUUUUUU", "UUUUUUUUR", "RTUUUUUUUU"} {
		steps := make([]string, 60)
		for i := range steps {
			steps[i] = shape
		}
		points := movingPoints(t, steps)
		blocks := 0
		for _, msg := range growingHistory(steps) {
			blocks += len(msg.Content)
		}
		require.Less(t, blocks-1-points[len(points)-1], 20, "%s: the point fell behind the history", shape)
	}
}
