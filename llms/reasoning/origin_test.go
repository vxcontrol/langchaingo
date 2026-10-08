package reasoning

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestModelFamiliesAreReadUnderEveryPrefix(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"claude-sonnet-4-5", "anthropic/claude-opus-5-5", "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
		"bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", "vertex_ai/claude-sonnet-4-5@20250929",
	} {
		require.True(t, IsClaude(model), model)
		require.False(t, IsGemini(model), model)
	}
	for _, model := range []string{"gemini-2.5-flash", "vertex_ai/gemini-3-pro-preview", "openrouter/google/gemini-2.5-pro"} {
		require.True(t, IsGemini(model), model)
		require.False(t, IsClaude(model), model)
	}
	for _, model := range []string{"gpt-5", "deepseek-reasoner", "gemma-3-27b-it", "my-alias", ""} {
		require.False(t, IsClaude(model), model)
		require.False(t, IsGemini(model), model)
	}
}

func TestClaudeGetsOnlySignedReasoningNoOtherModelWrote(t *testing.T) {
	t.Parallel()

	const target = "claude-sonnet-4-5"
	signed := func(model string) *ContentReasoning {
		return FromBlocks([]Block{{Text: "plan", Signature: []byte("sig")}}).WrittenBy(model)
	}
	mixed := FromBlocks([]Block{
		{Text: "unsigned"}, {Text: "plan", Signature: []byte("sig")}, {Redacted: []byte("opaque"), AfterToolCalls: 1},
	})

	require.Equal(t, signed("claude-opus-4-8"), ForClaude(signed("claude-opus-4-8"), target), "another Claude model")
	require.Equal(t, signed(""), ForClaude(signed(""), target), "a writer that was never recorded")
	require.Equal(t, signed("my-alias"), ForClaude(signed("my-alias"), "my-alias"), "the target itself")
	require.Nil(t, ForClaude(signed("gemini-2.5-pro"), target))
	require.Nil(t, ForClaude(signed("deepseek-reasoner"), target))
	require.Nil(t, ForClaude(signed("pentagi-sonnet"), target), "a writer whose name does not say Claude")
	require.Nil(t, ForClaude(&ContentReasoning{Content: "plan"}, target), "no signature")
	require.Equal(t,
		[]Block{{Text: "plan", Signature: []byte("sig")}, {Redacted: []byte("opaque"), AfterToolCalls: 1}},
		ForClaude(mixed, target).Sequence())
}

func TestGeminiGetsNoSignatureAnotherModelWrote(t *testing.T) {
	t.Parallel()

	const target = "gemini-2.5-flash"
	signature := func(model string) *ContentReasoning {
		return (&ContentReasoning{Signature: []byte("sig")}).WrittenBy(model)
	}

	require.Equal(t, signature("gemini-2.5-pro"), ForGemini(signature("gemini-2.5-pro"), target))
	require.Equal(t, signature(""), ForGemini(signature(""), target))
	require.Nil(t, ForGemini(signature("claude-sonnet-4-5"), target))
	require.Nil(t, ForGemini(signature("gpt-5"), target))
}

func TestTheWriterSurvivesStorage(t *testing.T) {
	t.Parallel()

	written := FromBlocks([]Block{{Text: "plan", Signature: []byte("sig")}}).WrittenBy("claude-sonnet-4-5")
	data, err := json.Marshal(written)
	require.NoError(t, err)
	var read ContentReasoning
	require.NoError(t, json.Unmarshal(data, &read))
	require.Equal(t, written, &read)

	var legacy ContentReasoning
	require.NoError(t, json.Unmarshal([]byte(`{"content":"plan","model":"claude-sonnet-4-5","redacted":"b3BhcXVl"}`), &legacy))
	require.Equal(t, "claude-sonnet-4-5", legacy.Model)
	require.Equal(t, []Block{{Text: "plan"}, {Redacted: []byte("opaque")}}, legacy.Sequence())
}
