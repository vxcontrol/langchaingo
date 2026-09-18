package llms_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestAChainStoredByEarlierBuildsStillReads(t *testing.T) {
	t.Parallel()

	for name, stored := range map[string]string{
		"encrypted thought as one string": `[
			{"role":"human","text":"hi"},
			{"role":"ai","parts":[
				{"type":"text","text":"x","reasoning":{"content":"t","signature":"c2ln","redacted":"cmVk"}},
				{"type":"tool_call","tool_call":{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"},"reasoning":{"redacted":"cmVk"}}}
			]}
		]`,
		"encrypted thoughts as an array": `[
			{"role":"human","text":"hi"},
			{"role":"ai","parts":[
				{"type":"text","text":"x","reasoning":{"content":"t","signature":"c2ln","redacted":["cmVk"]}},
				{"type":"tool_call","tool_call":{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"},"reasoning":{"redacted":["cmVk"]}}}
			]}
		]`,
	} {
		var chain []llms.MessageContent
		require.NoError(t, json.Unmarshal([]byte(stored), &chain), name)
		require.Len(t, chain, 2, name)
		require.Len(t, chain[1].Parts, 2, name)

		text, ok := chain[1].Parts[0].(llms.TextContent)
		require.True(t, ok, name)
		assert.Equal(t,
			[]reasoning.Block{{Text: "t", Signature: []byte("sig")}, {Redacted: []byte("red")}},
			text.Reasoning.Sequence(), name)

		call, ok := chain[1].Parts[1].(llms.ToolCall)
		require.True(t, ok, name)
		assert.Equal(t, []reasoning.Block{{Redacted: []byte("red")}}, call.Reasoning.Sequence(), name)
	}
}

func TestAChainKeepsTheBlocksOfEveryPartThroughStorage(t *testing.T) {
	t.Parallel()

	thought := reasoning.FromBlocks([]reasoning.Block{
		{Text: "a", Signature: []byte("s1")},
		{Signature: []byte("s2"), AfterToolCalls: 1},
	})
	sent := []llms.MessageContent{
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextPartWithReasoning("", thought),
			llms.ToolCall{ID: "c1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "f", Arguments: "{}"}},
			llms.ToolCall{ID: "c2", Type: "function", FunctionCall: &llms.FunctionCall{Name: "g", Arguments: "{}"}},
		}},
	}

	data, err := json.Marshal(sent)
	require.NoError(t, err)

	var got []llms.MessageContent
	require.NoError(t, json.Unmarshal(data, &got))
	text, ok := got[0].Parts[0].(llms.TextContent)
	require.True(t, ok)
	assert.Equal(t, thought.Sequence(), text.Reasoning.Sequence())
}
