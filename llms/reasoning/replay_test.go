package reasoning

import (
	"testing"

	"github.com/stretchr/testify/require"
)

const (
	claudeAPI = "api.anthropic.com"
	gateway   = "llm.pentagi.net"
	dashscope = "dashscope-intl.aliyuncs.com"
)

func on(model, host string, api ReplayAPI) ReplayTarget {
	return ReplayTarget{Model: model, Host: host, API: api, Tools: true, Mode: ThinkingAdaptive}
}

func withoutTools(t ReplayTarget) ReplayTarget { t.Tools = false; return t }
func onBudget(t ReplayTarget) ReplayTarget     { t.Mode = ThinkingBudget; return t }
func thinkingOff(t ReplayTarget) ReplayTarget  { t.Mode = ThinkingOff; return t }
func keeping(t ReplayTarget) ReplayTarget      { t.KeepsPastReasoning = true; return t }

func TestTheReplayPolicyFollowsTheTargetModelAndItsHost(t *testing.T) { //nolint:funlen
	t.Parallel()

	appendOnly := Replay{Binding: BindingPrefix, ChecksPrefix: true, Needs: PastReasoningOpenLoop,
		OwnLoop: LoopWithoutThinking, ForeignLoop: LoopWithoutThinking}
	claude := Replay{Needs: PastReasoningOpenLoop, OwnLoop: LoopAsSent, ForeignLoop: LoopWithoutThinking}
	claudeOnBudget := Replay{Needs: PastReasoningOpenLoop, HostDropsPast: true, OwnLoop: LoopAsSent, ForeignLoop: LoopInSummary}
	converse := Replay{Binding: BindingAllMessages, Needs: PastReasoningOpenLoop, HostDropsPast: true,
		OwnLoop: LoopInSummary, ForeignLoop: LoopInSummary}
	converseAppendOnly := Replay{Binding: BindingAllMessages, ChecksPrefix: true, Needs: PastReasoningOpenLoop,
		OwnLoop: LoopInSummary, ForeignLoop: LoopInSummary}
	gateway5 := Replay{Needs: PastReasoningOpenLoop, OwnLoop: LoopAsSent, ForeignLoop: LoopInSummary}
	everyTurn := Replay{Needs: PastReasoningEveryTurn, OwnLoop: LoopAsSent, ForeignLoop: LoopInSummary}
	dropsPast := Replay{HostDropsPast: true, OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}
	inLoop := Replay{Needs: PastReasoningOpenLoop, HostDropsPast: true, OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}
	asSent := Replay{OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}

	for _, tc := range []struct {
		name   string
		target ReplayTarget
		want   Replay
	}{
		{"Claude that checks the prefix", on("claude-opus-5-5", claudeAPI, ReplayMessages), appendOnly},
		{"its dated snapshot", on("claude-sonnet-5-5-20260915", claudeAPI, ReplayMessages), appendOnly},
		{"through a gateway's passthrough", on("anthropic/claude-fable-5-1", gateway, ReplayMessages), appendOnly},
		{"with its thinking off", thinkingOff(on("claude-haiku-5-5", claudeAPI, ReplayMessages)), appendOnly},
		{"without tools", withoutTools(on("claude-opus-5-5", claudeAPI, ReplayMessages)),
			Replay{Binding: BindingPrefix, ChecksPrefix: true, OwnLoop: LoopWithoutThinking, ForeignLoop: LoopWithoutThinking}},
		{"Claude that checks no prefix", on("claude-sonnet-5", claudeAPI, ReplayMessages), claude},
		{"Mythos 5.1 checks no prefix", on("claude-mythos-5-1", claudeAPI, ReplayMessages), claude},
		{"Opus 4.8", on("claude-opus-4-8", claudeAPI, ReplayMessages), claude},
		{"Opus 4.7", on("claude-opus-4-7", claudeAPI, ReplayMessages), claude},
		{"Sonnet 4.6", on("claude-sonnet-4-6", claudeAPI, ReplayMessages), claude},
		{"Mythos Preview", on("claude-mythos-preview", claudeAPI, ReplayMessages), claude},
		{"Opus 4.5 keeps every turn on a budget", onBudget(on("claude-opus-4-5", claudeAPI, ReplayMessages)),
			Replay{Needs: PastReasoningOpenLoop, OwnLoop: LoopAsSent, ForeignLoop: LoopInSummary}},
		{"Opus 4.1 keeps the last turn", onBudget(on("claude-opus-4-1", claudeAPI, ReplayMessages)), claudeOnBudget},
		{"Opus 4.6 thinking adaptively", on("claude-opus-4-6", claudeAPI, ReplayMessages), claude},
		{"Opus 4.6 on a budget", onBudget(on("claude-opus-4-6", claudeAPI, ReplayMessages)),
			Replay{Needs: PastReasoningOpenLoop, OwnLoop: LoopAsSent, ForeignLoop: LoopInSummary}},
		{"Claude on a budget", onBudget(on("claude-sonnet-4-5", claudeAPI, ReplayMessages)), claudeOnBudget},
		{"adaptive asked of a budget-only model", on("claude-haiku-4-5", claudeAPI, ReplayMessages), claudeOnBudget},
		{"Claude on a budget with thinking off", thinkingOff(on("claude-sonnet-4-5", claudeAPI, ReplayMessages)),
			Replay{Needs: PastReasoningOpenLoop, HostDropsPast: true, OwnLoop: LoopAsSent, ForeignLoop: LoopWithoutThinking}},
		{"Claude on Converse", onBudget(on("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "", ReplayConverse)), converse},
		{"Claude on Converse that checks the prefix", on("global.anthropic.claude-opus-5-5-v1:0", "", ReplayConverse), converseAppendOnly},
		{"Claude through a gateway's /v1", on("anthropic/claude-sonnet-5", gateway, ReplayChat), gateway5},
		{"Claude that checks the prefix through a gateway's /v1", on("anthropic/claude-opus-5-5", gateway, ReplayChat),
			Replay{Binding: BindingPrefix, ChecksPrefix: true, Needs: PastReasoningOpenLoop,
				OwnLoop: LoopWithoutThinking, ForeignLoop: LoopInSummary}},
		{"Claude through a gateway's Bedrock route", on("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", gateway, ReplayChat),
			converse},
		{"Claude through a gateway's Converse route", on("bedrock/converse/global.anthropic.claude-opus-5-5-v1:0", gateway, ReplayChat),
			converseAppendOnly},
		{"Claude through a gateway's InvokeModel route", on("bedrock/invoke/us.anthropic.claude-sonnet-5-v1:0", gateway, ReplayChat),
			gateway5},
		{"Claude on a public provider", on("anthropic/claude-opus-5-5", "openrouter.ai", ReplayChat), asSent},
		{"Gemini 3", on("gemini-3-pro-preview", "", ReplayGemini),
			Replay{Binding: BindingCurrentTurn, Needs: PastReasoningOpenLoop, HostDropsPast: true,
				OwnLoop: LoopWithoutThinking, ForeignLoop: LoopWithoutThinking}},
		{"Gemini 3 without tools", withoutTools(on("gemini-3-pro-preview", "", ReplayGemini)),
			Replay{Binding: BindingCurrentTurn, HostDropsPast: true, OwnLoop: LoopWithoutThinking, ForeignLoop: LoopWithoutThinking}},
		{"Gemini 3.5", withoutTools(on("gemini-3.5-flash", "", ReplayGemini)),
			Replay{Binding: BindingCurrentTurn, Needs: PastReasoningEveryTurn, OwnLoop: LoopWithoutThinking, ForeignLoop: LoopWithoutThinking}},
		{"Gemini 2.5", on("gemini-2.5-flash", "", ReplayGemini), asSent},
		{"DeepSeek with tools", on("deepseek-v4-pro", "api.deepseek.com", ReplayChat), everyTurn},
		{"DeepSeek through a gateway's route", on("deepseek/deepseek-v4-flash", gateway, ReplayChat), everyTurn},
		{"DeepSeek on its Anthropic endpoint", on("deepseek-v4-pro", "api.deepseek.com", ReplayMessages), everyTurn},
		{"DeepSeek without tools", withoutTools(on("deepseek-v4-pro", "api.deepseek.com", ReplayChat)), dropsPast},
		{"DeepSeek with its thinking off", thinkingOff(on("deepseek-flash", "api.deepseek.com", ReplayChat)), asSent},
		{"DeepSeek on another vendor's host", on("deepseek-v4-pro", dashscope, ReplayChat), asSent},
		{"Kimi K3", withoutTools(on("kimi-k3", "api.moonshot.ai", ReplayChat)), everyTurn},
		{"Kimi K3 through a gateway's route", on("moonshot/kimi-k3", gateway, ReplayChat), everyTurn},
		{"Kimi K2.7 Code always thinks", thinkingOff(on("kimi-k2.7-code", "api.moonshot.ai", ReplayChat)), everyTurn},
		{"Kimi K2.6 in a tool loop", on("kimi-k2.6", "api.moonshot.ai", ReplayChat), inLoop},
		{"Kimi K2.6 without tools", withoutTools(on("kimi-k2.6", "api.moonshot.ai", ReplayChat)), dropsPast},
		{"Kimi K2.6 keeping its thinking", keeping(on("kimi-k2.6", "api.moonshot.ai", ReplayChat)), everyTurn},
		{"Kimi K2.6 with its thinking off", thinkingOff(on("kimi-k2.6", "api.moonshot.ai", ReplayChat)), asSent},
		{"Kimi on Bedrock", on("moonshotai.kimi-k3", "", ReplayConverse), asSent},
		{"MiniMax with tools", on("MiniMax-M3", "api.minimax.io", ReplayChat), everyTurn},
		{"MiniMax through a gateway's route", on("minimax/MiniMax-M3", gateway, ReplayChat), everyTurn},
		{"MiniMax without tools", withoutTools(on("MiniMax-M3", "api.minimax.io", ReplayChat)), asSent},
		{"Qwen 3.8", on("qwen3.8-max", dashscope, ReplayChat), everyTurn},
		{"Qwen 3.8 through a gateway's route", on("dashscope/qwen3.8-max", gateway, ReplayChat), everyTurn},
		{"Qwen 3.8 with its thinking off", thinkingOff(on("qwen3.8-max", dashscope, ReplayChat)), asSent},
		{"a Qwen 3.8 that preserves nothing", on("qwen3.8-27b", dashscope, ReplayChat), asSent},
		{"Qwen 3.7", on("qwen3.7-plus", dashscope, ReplayChat), dropsPast},
		{"Qwen 3.7 keeping its thinking", keeping(on("qwen3.7-plus", dashscope, ReplayChat)), everyTurn},
		{"Kimi K2.7 Code on DashScope", on("kimi-k2.7-code", dashscope, ReplayChat), everyTurn},
		{"GLM 5.2 on DashScope", on("glm-5.2", dashscope, ReplayChat), everyTurn},
		{"GLM 5.3 on DashScope", on("glm-5.3", dashscope, ReplayChat), dropsPast},
		{"GLM 5.3 on DashScope keeping its thinking", keeping(on("glm-5.3", dashscope, ReplayChat)), everyTurn},
		{"GLM in a tool loop", on("glm-5.3", "api.z.ai", ReplayChat), inLoop},
		{"GLM keeping its thinking", keeping(on("glm-5.3", "api.z.ai", ReplayChat)), everyTurn},
		{"GLM through a gateway's route", keeping(on("zai/glm-5.3", gateway, ReplayChat)), everyTurn},
		{"GLM with its thinking off", thinkingOff(keeping(on("glm-5.3", "api.z.ai", ReplayChat))), asSent},
		{"Mistral's reasoning model", on("magistral-medium-latest", "api.mistral.ai", ReplayChat), everyTurn},
		{"Mistral through a gateway's route", on("mistral/magistral-medium-latest", gateway, ReplayChat), everyTurn},
		{"Mistral without reasoning", on("mistral-large-latest", "api.mistral.ai", ReplayChat), asSent},
		{"Mistral with its thinking off", thinkingOff(on("magistral-medium-latest", "api.mistral.ai", ReplayChat)), asSent},
		{"Grok", on("grok-4.7", "api.x.ai", ReplayChat), Replay{Needs: PastReasoningOwnTurns, OwnLoop: LoopAsSent, ForeignLoop: LoopAsSent}},
		{"GPT", on("gpt-5.6", "api.openai.com", ReplayChat), asSent},
		{"Ollama", on("deepseek-v4-pro", "localhost", ReplayOllama), asSent},
		{"a name outside every line", on("my-model", gateway, ReplayChat), asSent},
	} {
		require.Equal(t, tc.want, ReplayPolicy(tc.target), tc.name)
	}
}

func TestAnUnlistedVersionReplaysAsTheReleaseItFollows(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		unlisted, listed, host string
		api                    ReplayAPI
	}{
		{"claude-opus-5-6", "claude-opus-5-5", claudeAPI, ReplayMessages},
		{"kimi-k3.1", "kimi-k3", "api.moonshot.ai", ReplayChat},
		{"qwen3.9-max", "qwen3.8-max", dashscope, ReplayChat},
		{"gemini-3.9-flash", "gemini-3.8-flash", "", ReplayGemini},
		{"glm-5.4", "glm-5.3", "api.z.ai", ReplayChat},
	} {
		got := ReplayPolicy(on(tc.unlisted, tc.host, tc.api))
		require.Equal(t, tc.listed, got.Inherits, tc.unlisted)
		got.Inherits = ""
		require.Equal(t, ReplayPolicy(on(tc.listed, tc.host, tc.api)), got, tc.unlisted)
	}
	require.Empty(t, ReplayPolicy(on("claude-opus-5-5", claudeAPI, ReplayMessages)).Inherits)
}

func TestAnExecutorChangeNeedsABoundaryOnlyWhereTheReaderCannotContinue(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		reader ReplayTarget
		kin    string
	}{
		{on("deepseek-v4-pro", "api.deepseek.com", ReplayChat), "deepseek-v4-flash"},
		{on("kimi-k3", "api.moonshot.ai", ReplayChat), "kimi-k2.7-code"},
		{on("MiniMax-M3", "api.minimax.io", ReplayChat), "minimax-m3"},
		{on("qwen3.8-max", dashscope, ReplayChat), "qwen3.8-flash"},
		{keeping(on("glm-5.3", "api.z.ai", ReplayChat)), "glm-5.2"},
		{on("magistral-medium-latest", "api.mistral.ai", ReplayChat), "magistral-small-latest"},
		{on("gemini-3.5-flash", "", ReplayGemini), "gemini-3.5-pro"},
	} {
		require.True(t, NeedsBoundary(tc.reader, "claude-opus-5-5", false), "%s after another vendor", tc.reader.Model)
		require.True(t, NeedsBoundary(tc.reader, "pentagi-alias", false), "%s after a name that names no vendor", tc.reader.Model)
		require.False(t, NeedsBoundary(tc.reader, tc.kin, false), "%s after %s", tc.reader.Model, tc.kin)
		require.False(t, NeedsBoundary(tc.reader, "", false), "%s after a writer that was never recorded", tc.reader.Model)
	}

	require.False(t, NeedsBoundary(on("grok-4.7", "api.x.ai", ReplayChat), "claude-opus-5-5", true),
		"another model's answers cost Grok's cache nothing")

	onBudgetReader := onBudget(on("claude-sonnet-4-5", claudeAPI, ReplayMessages))
	require.True(t, NeedsBoundary(onBudgetReader, "claude-opus-4-8", true), "an open loop whose thinking another model wrote")
	require.True(t, NeedsBoundary(onBudgetReader, "gemini-3-pro-preview", true))
	require.False(t, NeedsBoundary(onBudgetReader, "claude-sonnet-4-5-20250929", true), "its own open loop")
	require.False(t, NeedsBoundary(onBudgetReader, "claude-opus-4-8", false), "no open loop")
	require.True(t, NeedsBoundary(onBudget(on("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "", ReplayConverse)), "claude-opus-4-8", true),
		"on Converse")

	adaptive := on("claude-opus-5-5", claudeAPI, ReplayMessages)
	require.False(t, NeedsBoundary(adaptive, "claude-fable-5-1", true), "the API drops the blocks the model cannot read")
	require.False(t, NeedsBoundary(adaptive, "gemini-3-pro-preview", true))

	require.False(t, NeedsBoundary(on("gpt-5.6", "api.openai.com", ReplayChat), "claude-opus-5-5", true))
}
