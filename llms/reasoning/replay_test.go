package reasoning

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTheReplayPolicyFollowsTheTargetModelAndItsHost(t *testing.T) {
	t.Parallel()

	const (
		claudeAPI = "api.anthropic.com"
		gateway   = "llm.pentagi.net"
	)
	appendOnly := Replay{Binding: BindingPrefix, ChecksPrefix: true, Needs: PastReasoningOpenLoop,
		OwnLoop: LoopWithoutThinking, ForeignLoop: LoopWithoutThinking}
	claude := Replay{Needs: PastReasoningOpenLoop, OwnLoop: LoopAsSent, ForeignLoop: LoopWithoutThinking}
	claudeOnBudget := Replay{Needs: PastReasoningOpenLoop, HostDropsPast: true, OwnLoop: LoopAsSent, ForeignLoop: LoopInSummary}
	everyTurn := Replay{Needs: PastReasoningEveryTurn, OwnLoop: LoopAsSent, ForeignLoop: LoopInSummary}
	dropsPast := Replay{HostDropsPast: true, OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}
	asSent := Replay{OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}

	for _, tc := range []struct {
		name  string
		model string
		host  string
		api   ReplayAPI
		tools bool
		mode  ThinkingMode
		opts  ReplayOptions
		want  Replay
	}{
		{"Claude that checks the prefix", "claude-opus-5-5", claudeAPI, ReplayMessages, true, ThinkingAdaptive, ReplayOptions{}, appendOnly},
		{"its dated snapshot", "claude-sonnet-5-5-20260915", claudeAPI, ReplayMessages, true, ThinkingAdaptive, ReplayOptions{}, appendOnly},
		{"through a gateway's passthrough", "anthropic/claude-fable-5-1", gateway, ReplayMessages, true, ThinkingAdaptive, ReplayOptions{}, appendOnly},
		{"with its thinking off", "claude-haiku-5-5", claudeAPI, ReplayMessages, true, ThinkingOff, ReplayOptions{}, appendOnly},
		{"without tools", "claude-opus-5-5", claudeAPI, ReplayMessages, false, ThinkingAdaptive, ReplayOptions{},
			Replay{Binding: BindingPrefix, ChecksPrefix: true, OwnLoop: LoopWithoutThinking, ForeignLoop: LoopWithoutThinking}},
		{"Claude that checks no prefix", "claude-sonnet-5", claudeAPI, ReplayMessages, true, ThinkingAdaptive, ReplayOptions{}, claude},
		{"Mythos 5.1 checks no prefix", "claude-mythos-5-1", claudeAPI, ReplayMessages, true, ThinkingAdaptive, ReplayOptions{}, claude},
		{"Claude on a budget", "claude-sonnet-4-5", claudeAPI, ReplayMessages, true, ThinkingBudget, ReplayOptions{}, claudeOnBudget},
		{"adaptive asked of a budget-only model", "claude-haiku-4-5", claudeAPI, ReplayMessages, true, ThinkingAdaptive, ReplayOptions{},
			claudeOnBudget},
		{"Claude on a budget with thinking off", "claude-sonnet-4-5", claudeAPI, ReplayMessages, true, ThinkingOff, ReplayOptions{},
			Replay{Needs: PastReasoningOpenLoop, HostDropsPast: true, OwnLoop: LoopAsSent, ForeignLoop: LoopWithoutThinking}},
		{"Claude on Converse", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "", ReplayConverse, true, ThinkingBudget, ReplayOptions{},
			Replay{Binding: BindingAllMessages, Needs: PastReasoningOpenLoop, HostDropsPast: true, OwnLoop: LoopInSummary, ForeignLoop: LoopInSummary}},
		{"Claude on Converse that checks the prefix", "global.anthropic.claude-opus-5-5-v1:0", "", ReplayConverse, true, ThinkingAdaptive,
			ReplayOptions{}, Replay{Binding: BindingAllMessages, ChecksPrefix: true, Needs: PastReasoningOpenLoop,
				OwnLoop: LoopInSummary, ForeignLoop: LoopInSummary}},
		{"Claude through a gateway's /v1", "anthropic/claude-sonnet-5", gateway, ReplayChat, true, ThinkingAdaptive, ReplayOptions{},
			Replay{Needs: PastReasoningOpenLoop, OwnLoop: LoopAsSent, ForeignLoop: LoopInSummary}},
		{"Claude that checks the prefix through a gateway's /v1", "anthropic/claude-opus-5-5", gateway, ReplayChat, true, ThinkingAdaptive,
			ReplayOptions{}, Replay{Binding: BindingPrefix, ChecksPrefix: true, Needs: PastReasoningOpenLoop,
				OwnLoop: LoopWithoutThinking, ForeignLoop: LoopInSummary}},
		{"Claude on a public provider", "anthropic/claude-opus-5-5", "openrouter.ai", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, asSent},
		{"Gemini 3", "gemini-3-pro-preview", "", ReplayGemini, true, ThinkingAdaptive, ReplayOptions{},
			Replay{Binding: BindingCurrentTurn, Needs: PastReasoningOpenLoop, OwnLoop: LoopWithoutThinking, ForeignLoop: LoopWithoutThinking}},
		{"Gemini 2.5", "gemini-2.5-flash", "", ReplayGemini, true, ThinkingAdaptive, ReplayOptions{}, asSent},
		{"DeepSeek with tools", "deepseek-v4-pro", "api.deepseek.com", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, everyTurn},
		{"DeepSeek through a gateway's route", "deepseek/deepseek-v4-flash", gateway, ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, everyTurn},
		{"DeepSeek on its Anthropic endpoint", "deepseek-v4-pro", "api.deepseek.com", ReplayMessages, true, ThinkingAdaptive, ReplayOptions{},
			everyTurn},
		{"DeepSeek without tools", "deepseek-v4-pro", "api.deepseek.com", ReplayChat, false, ThinkingAdaptive, ReplayOptions{}, dropsPast},
		{"DeepSeek on another vendor's host", "deepseek-v4-pro", "dashscope-intl.aliyuncs.com", ReplayChat, true, ThinkingAdaptive,
			ReplayOptions{}, asSent},
		{"Kimi K3", "kimi-k3", "api.moonshot.ai", ReplayChat, false, ThinkingAdaptive, ReplayOptions{}, everyTurn},
		{"Kimi K2.7 Code", "kimi-k2.7-code", "api.moonshot.ai", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, everyTurn},
		{"Kimi K2.6", "kimi-k2.6", "api.moonshot.ai", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, dropsPast},
		{"Kimi K2.6 keeping its thinking", "kimi-k2.6", "api.moonshot.ai", ReplayChat, true, ThinkingAdaptive,
			ReplayOptions{KeepsPastReasoning: true}, everyTurn},
		{"Kimi on Bedrock", "moonshotai.kimi-k3", "", ReplayConverse, true, ThinkingAdaptive, ReplayOptions{}, asSent},
		{"MiniMax with tools", "MiniMax-M3", "api.minimax.io", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, everyTurn},
		{"MiniMax without tools", "MiniMax-M3", "api.minimax.io", ReplayChat, false, ThinkingAdaptive, ReplayOptions{}, asSent},
		{"Qwen 3.8", "qwen3.8-max", "dashscope-intl.aliyuncs.com", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, everyTurn},
		{"Qwen 3.7", "qwen3.7-plus", "dashscope-intl.aliyuncs.com", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, asSent},
		{"GLM keeping its thinking", "glm-5.3", "api.z.ai", ReplayChat, true, ThinkingAdaptive, ReplayOptions{KeepsPastReasoning: true}, everyTurn},
		{"GLM", "glm-5.3", "api.z.ai", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, dropsPast},
		{"Mistral's reasoning model", "magistral-medium-latest", "api.mistral.ai", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, everyTurn},
		{"Grok", "grok-4.7", "api.x.ai", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, asSent},
		{"GPT", "gpt-5.6", "api.openai.com", ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, asSent},
		{"Ollama", "deepseek-v4-pro", "localhost:11434", ReplayOllama, true, ThinkingAdaptive, ReplayOptions{}, asSent},
		{"a name outside every line", "my-model", gateway, ReplayChat, true, ThinkingAdaptive, ReplayOptions{}, asSent},
	} {
		got := ReplayPolicy(tc.model, tc.host, tc.api, tc.tools, tc.mode, tc.opts)
		got.reader = readerOf{}
		require.Equal(t, tc.want, got, tc.name)
	}
}

func TestAnUnlistedVersionReplaysAsTheReleaseItFollows(t *testing.T) {
	t.Parallel()

	got := ReplayPolicy("claude-opus-5-6", "api.anthropic.com", ReplayMessages, true, ThinkingAdaptive, ReplayOptions{})
	require.Equal(t, "claude-opus-5-5", got.Inherits)
	require.True(t, got.ChecksPrefix)

	require.Empty(t, ReplayPolicy("claude-opus-5-5", "api.anthropic.com", ReplayMessages, true, ThinkingAdaptive, ReplayOptions{}).Inherits)
}

func TestAnExecutorChangeNeedsABoundaryOnlyWhereTheReaderCannotContinue(t *testing.T) {
	t.Parallel()

	deepseek := ReplayPolicy("deepseek-v4-pro", "api.deepseek.com", ReplayChat, true, ThinkingAdaptive, ReplayOptions{})
	require.True(t, deepseek.NeedsBoundaryAfter("claude-opus-5-5", false), "another vendor's turns lack the reasoning DeepSeek requires")
	require.True(t, deepseek.NeedsBoundaryAfter("pentagi-alias", false), "a writer whose name names no vendor")
	require.False(t, deepseek.NeedsBoundaryAfter("deepseek-v4-flash", false))
	require.False(t, deepseek.NeedsBoundaryAfter("", false), "a writer that was never recorded")

	onBudget := ReplayPolicy("claude-sonnet-4-5", "api.anthropic.com", ReplayMessages, true, ThinkingBudget, ReplayOptions{})
	require.True(t, onBudget.NeedsBoundaryAfter("claude-opus-4-8", true), "an open loop whose thinking another model wrote")
	require.True(t, onBudget.NeedsBoundaryAfter("gemini-3-pro-preview", true))
	require.False(t, onBudget.NeedsBoundaryAfter("claude-sonnet-4-5-20250929", true), "its own open loop")
	require.False(t, onBudget.NeedsBoundaryAfter("claude-opus-4-8", false), "no open loop")

	adaptive := ReplayPolicy("claude-opus-5-5", "api.anthropic.com", ReplayMessages, true, ThinkingAdaptive, ReplayOptions{})
	require.False(t, adaptive.NeedsBoundaryAfter("claude-fable-5-1", true), "the API drops the blocks the model cannot read")
	require.False(t, adaptive.NeedsBoundaryAfter("gemini-3-pro-preview", true))

	gpt := ReplayPolicy("gpt-5.6", "api.openai.com", ReplayChat, true, ThinkingAdaptive, ReplayOptions{})
	require.False(t, gpt.NeedsBoundaryAfter("claude-opus-5-5", true))
}
