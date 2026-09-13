package reasoning

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestALonePlainBlockKeepsTheClassicShape(t *testing.T) {
	t.Parallel()

	got := FromBlocks([]Block{{Text: "step", Signature: []byte("sig")}})

	require.NotNil(t, got)
	assert.Equal(t, "step", got.Content)
	assert.Equal(t, []byte("sig"), got.Signature)
	assert.Empty(t, got.Blocks)
	assert.Equal(t, []Block{{Text: "step", Signature: []byte("sig")}}, got.Sequence())
}

func TestEveryBlockKeepsItsOwnSignature(t *testing.T) {
	t.Parallel()

	blocks := []Block{
		{Text: "first", Signature: []byte("s1")},
		{Text: "second", Signature: []byte("s2")},
	}
	got := FromBlocks(blocks)

	require.NotNil(t, got)
	assert.Equal(t, blocks, got.Sequence())
	assert.Equal(t, "firstsecond", got.Content, "Content mirrors the readable text of every block")
	assert.Empty(t, got.Signature, "no single signature covers two blocks")
}

func TestALoneBlockThatIsEncryptedOrPlacedStaysABlock(t *testing.T) {
	t.Parallel()

	for name, block := range map[string]Block{
		"encrypted":            {Redacted: []byte("opaque")},
		"after a tool call":    {Text: "update", Signature: []byte("s"), AfterToolCalls: 1},
		"encrypted and placed": {Redacted: []byte("opaque"), AfterToolCalls: 2},
	} {
		got := FromBlocks([]Block{block})
		require.NotNil(t, got, name)
		assert.Equal(t, []Block{block}, got.Blocks, name)
		assert.Empty(t, got.Signature, name)
	}
	assert.Nil(t, FromBlocks(nil))
}

func TestAHandBuiltReasoningIsOneBlock(t *testing.T) {
	t.Parallel()

	assert.Equal(t,
		[]Block{{Text: "step", Signature: []byte("sig")}},
		(&ContentReasoning{Content: "step", Signature: []byte("sig")}).Sequence())
	assert.Equal(t,
		[]Block{{Signature: []byte("sig")}},
		(&ContentReasoning{Signature: []byte("sig")}).Sequence(),
		"a signature with its text withheld is still a block to return")
	assert.Equal(t,
		[]Block{{Text: "step", Signature: []byte("sig")}, {Redacted: []byte("r1")}, {Redacted: []byte("r2")}},
		(&ContentReasoning{Content: "step", Signature: []byte("sig"), Redacted: [][]byte{[]byte("r1"), []byte("r2")}}).Sequence())
	assert.Empty(t, (&ContentReasoning{}).Sequence())
	assert.Empty(t, (*ContentReasoning)(nil).Sequence())
}

func TestBlocksAloneAreSomethingToCarryBack(t *testing.T) {
	t.Parallel()

	r := &ContentReasoning{Blocks: []Block{{Redacted: []byte("opaque")}, {Signature: []byte("s")}}}

	assert.False(t, r.IsEmpty())
	assert.False(t, r.HasContent())
	assert.Contains(t, r.String(), "Blocks: 2, 1 encrypted (6 bytes)")
}

func TestBlocksGroupAfterTheToolCallsAheadOfThem(t *testing.T) {
	t.Parallel()

	lead := Block{Text: "reasoning", Signature: []byte("s1")}
	update := Block{Text: "update", Signature: []byte("s2")}
	second := Block{Text: "between", Signature: []byte("s3"), AfterToolCalls: 1}
	trailing := Block{Text: "cut off", Signature: []byte("s4"), AfterToolCalls: 2}
	stray := Block{Text: "past the end", Signature: []byte("s5"), AfterToolCalls: 7}
	negative := Block{Text: "before anything", Signature: []byte("s6"), AfterToolCalls: -1}

	groups := GroupByToolCalls([]Block{lead, update, second, trailing, stray, negative}, 2)

	require.Len(t, groups, 3)
	assert.Equal(t, []Block{lead, update, negative}, groups[0])
	assert.Equal(t, []Block{second}, groups[1])
	assert.Equal(t, []Block{trailing, stray}, groups[2])
	assert.Equal(t, [][]Block{{lead, second}}, GroupByToolCalls([]Block{lead, second}, 0),
		"with no tool calls every block opens the turn")
}

func TestTheCollectorNotesTheToolCallsAheadOfEachBlock(t *testing.T) {
	t.Parallel()

	var c Collector
	c.Thought("reasoning", []byte("s1"))
	c.Thought("", nil)
	c.ToolCall()
	c.Encrypted(nil)
	c.Encrypted([]byte("opaque"))
	c.Thought("", []byte("s2"))
	c.ToolCall()

	got := c.Reasoning()

	require.NotNil(t, got)
	assert.Equal(t, []Block{
		{Text: "reasoning", Signature: []byte("s1")},
		{Redacted: []byte("opaque"), AfterToolCalls: 1},
		{Signature: []byte("s2"), AfterToolCalls: 1},
	}, got.Sequence())
	assert.Nil(t, (&Collector{}).Reasoning())
}

func TestStoredReasoningReadsInEveryShapeItWasWrittenIn(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name   string
		stored string
		want   []Block
	}{
		{
			name:   "content and signature",
			stored: `{"content":"step","signature":"c2ln"}`,
			want:   []Block{{Text: "step", Signature: []byte("sig")}},
		},
		{
			name:   "one encrypted block as a string",
			stored: `{"content":"step","signature":"c2ln","redacted":"cmVk"}`,
			want:   []Block{{Text: "step", Signature: []byte("sig")}, {Redacted: []byte("red")}},
		},
		{
			name:   "encrypted blocks as an array",
			stored: `{"redacted":["cjE=","cjI="]}`,
			want:   []Block{{Redacted: []byte("r1")}, {Redacted: []byte("r2")}},
		},
		{
			name:   "an empty encrypted value",
			stored: `{"content":"step","redacted":null}`,
			want:   []Block{{Text: "step"}},
		},
		{
			name:   "blocks",
			stored: `{"content":"ab","blocks":[{"text":"a","signature":"czE="},{"redacted":"cmVk","after_tool_calls":1}]}`,
			want:   []Block{{Text: "a", Signature: []byte("s1")}, {Redacted: []byte("red"), AfterToolCalls: 1}},
		},
	} {
		var got ContentReasoning
		require.NoError(t, json.Unmarshal([]byte(tc.stored), &got), tc.name)
		assert.Equal(t, tc.want, got.Sequence(), tc.name)
	}
}

func TestBlocksSurviveARoundTrip(t *testing.T) {
	t.Parallel()

	sent := FromBlocks([]Block{
		{Text: "a", Signature: []byte("s1")},
		{Redacted: []byte("opaque")},
		{Signature: []byte("s2"), AfterToolCalls: 1},
	})

	data, err := json.Marshal(sent)
	require.NoError(t, err)

	var got ContentReasoning
	require.NoError(t, json.Unmarshal(data, &got))
	assert.Equal(t, sent.Sequence(), got.Sequence())
	assert.Equal(t, sent.Content, got.Content)
}
