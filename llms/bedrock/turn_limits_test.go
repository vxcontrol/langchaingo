package bedrock_test

import (
	"context"
	"errors"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

var turnLimitTools = llms.WithTools([]llms.Tool{{
	Type:     "function",
	Function: &llms.FunctionDefinition{Name: "calc", Parameters: map[string]any{"type": "object"}},
}})

func turnLimitMessages(last llms.ChatMessageType) []llms.MessageContent {
	msgs := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
	if last == llms.ChatMessageTypeAI {
		msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeAI, "half an "))
	}
	return msgs
}

func TestBedrockRefusesTheSameTurnsAsThePrimaryDoor(t *testing.T) {
	t.Parallel()

	for _, converse := range []bool{false, true} {
		name := "invoke-model"
		opts := []bedrock.Option{}
		if converse {
			name = "converse"
			opts = append(opts, bedrock.WithConverseAPI())
		}

		t.Run(name+"/assistant prefill is refused before the request", func(t *testing.T) {
			t.Parallel()
			llm := truncationLLMWithBody(t, `{}`,
				append([]bedrock.Option{bedrock.WithModel("us.anthropic.claude-opus-4-6-v1:0")}, opts...)...)
			_, err := llm.GenerateContent(context.Background(), turnLimitMessages(llms.ChatMessageTypeAI))
			var target *reasoning.ErrAssistantPrefillUnsupported
			if !errors.As(err, &target) {
				t.Errorf("want ErrAssistantPrefillUnsupported, got %v", err)
			}
		})

		t.Run(name+"/assistant prefill behind a system message is refused before the request", func(t *testing.T) {
			t.Parallel()
			llm := truncationLLMWithBody(t, `{}`,
				append([]bedrock.Option{bedrock.WithModel("us.anthropic.claude-opus-4-6-v1:0")}, opts...)...)
			messages := append(turnLimitMessages(llms.ChatMessageTypeAI), llms.TextParts(llms.ChatMessageTypeSystem, "be brief"))
			_, err := llm.GenerateContent(context.Background(), messages)
			var target *reasoning.ErrAssistantPrefillUnsupported
			if !errors.As(err, &target) {
				t.Errorf("the system message is sent apart from the turns, so they still end on the answer: %v", err)
			}
		})

		t.Run(name+"/a forced tool with manual thinking is refused", func(t *testing.T) {
			t.Parallel()
			llm := truncationLLMWithBody(t, `{}`,
				append([]bedrock.Option{bedrock.WithModel("us.anthropic.claude-sonnet-4-5-v1:0")}, opts...)...)
			_, err := llm.GenerateContent(context.Background(), turnLimitMessages(llms.ChatMessageTypeHuman),
				llms.WithReasoning(llms.ReasoningMedium, 2048),
				turnLimitTools, llms.WithToolChoice(llms.ToolChoice{Type: "any"}))
			var target *reasoning.ErrForcedToolUseWithThinking
			if !errors.As(err, &target) {
				t.Errorf("want ErrForcedToolUseWithThinking, got %v", err)
			}
		})

		t.Run(name+"/a forced tool with a manual budget is refused on a generation that also does adaptive", func(t *testing.T) {
			t.Parallel()
			llm := truncationLLMWithBody(t, `{}`,
				append([]bedrock.Option{bedrock.WithModel("us.anthropic.claude-opus-4-6-v1:0")}, opts...)...)
			_, err := llm.GenerateContent(context.Background(), turnLimitMessages(llms.ChatMessageTypeHuman),
				llms.WithReasoning(llms.ReasoningMedium, 2048),
				turnLimitTools, llms.WithToolChoice(llms.ToolChoice{Type: "any"}))
			var target *reasoning.ErrForcedToolUseWithThinking
			if !errors.As(err, &target) {
				t.Errorf("this door sends the budget itself, so the rule holds even off a budget-only generation: %v", err)
			}
		})

		t.Run(name+"/a forced tool with adaptive thinking is allowed", func(t *testing.T) {
			t.Parallel()
			llm := truncationLLMWithBody(t, `{}`,
				append([]bedrock.Option{bedrock.WithModel("us.anthropic.claude-sonnet-5-v1:0")}, opts...)...)
			_, err := llm.GenerateContent(context.Background(), turnLimitMessages(llms.ChatMessageTypeHuman),
				llms.WithAdaptiveReasoning(llms.ReasoningMedium),
				turnLimitTools, llms.WithToolChoice(llms.ToolChoice{Type: "any"}))
			var target *reasoning.ErrForcedToolUseWithThinking
			if errors.As(err, &target) {
				t.Errorf("adaptive thinking carries no manual budget, so the rule must not fire: %v", err)
			}
		})

		t.Run(name+"/a non-Claude family is not refused for the same turn", func(t *testing.T) {
			t.Parallel()
			llm := truncationLLMWithBody(t, `{}`,
				append([]bedrock.Option{bedrock.WithModel("us.amazon.nova-pro-v1:0")}, opts...)...)
			_, err := llm.GenerateContent(context.Background(), turnLimitMessages(llms.ChatMessageTypeHuman),
				llms.WithReasoning(llms.ReasoningMedium, 2048),
				turnLimitTools, llms.WithToolChoice(llms.ToolChoice{Type: "any"}))
			var target *reasoning.ErrForcedToolUseWithThinking
			if errors.As(err, &target) {
				t.Errorf("the Claude-only rule refused a family that never carries a thinking payload: %v", err)
			}
		})
	}
}

func TestABedrockForcedChoiceWithABudgetAndNoToolsGoesOutWithoutTheChoice(t *testing.T) {
	t.Parallel()

	for _, converse := range []bool{false, true} {
		opts := []bedrock.Option{bedrock.WithModel("us.anthropic.claude-sonnet-4-5-v1:0")}
		answer := legacyAnswer
		if converse {
			opts = append(opts, bedrock.WithConverseAPI())
			answer = converseAnswer
		}
		_, body := bedrockWarningsSending(t, answer, opts,
			llms.WithReasoning(llms.ReasoningMedium, 2048),
			llms.WithToolChoice(llms.ToolChoice{Type: "any"}))
		if _, sent := body["tool_choice"]; sent {
			t.Errorf("converse=%v: the legacy body carries a tool choice with no tools: %v", converse, body)
		}
		if _, sent := body["toolConfig"]; sent {
			t.Errorf("converse=%v: the converse body carries a tool config with no tools: %v", converse, body)
		}
	}
}

func TestBedrockJudgesThePrefillOnTheTurnsItSends(t *testing.T) {
	t.Parallel()

	for _, converse := range []bool{false, true} {
		opts := []bedrock.Option{}
		if converse {
			opts = append(opts, bedrock.WithConverseAPI())
		}
		human, answer := llms.TextParts(llms.ChatMessageTypeHuman, "hi"), llms.TextParts(llms.ChatMessageTypeAI, "half an ")
		sendsNoAnswer := [][]llms.MessageContent{
			{human, {Role: llms.ChatMessageTypeAI}, llms.TextParts(llms.ChatMessageTypeSystem, "be brief")},
			{human, {Role: llms.ChatMessageTypeAI}},
		}
		endsOnTheAnswer := [][]llms.MessageContent{{human, answer, {Role: llms.ChatMessageTypeHuman}}}
		if converse {
			sendsNoAnswer = append(sendsNoAnswer, []llms.MessageContent{human, llms.TextParts(llms.ChatMessageTypeAI, "")})
			endsOnTheAnswer = append(endsOnTheAnswer, []llms.MessageContent{human, answer, llms.TextParts(llms.ChatMessageTypeHuman, "")})
		}
		llm := truncationLLMWithBody(t, `{}`,
			append([]bedrock.Option{bedrock.WithModel("us.anthropic.claude-opus-4-6-v1:0")}, opts...)...)
		var target *reasoning.ErrAssistantPrefillUnsupported
		for i, messages := range sendsNoAnswer {
			if _, err := llm.GenerateContent(context.Background(), messages); errors.As(err, &target) {
				t.Errorf("converse=%v: conversation %d sends no assistant turn last: %v", converse, i, err)
			}
		}
		for i, messages := range endsOnTheAnswer {
			if _, err := llm.GenerateContent(context.Background(), messages); !errors.As(err, &target) {
				t.Errorf("converse=%v: conversation %d sends the answer last, want ErrAssistantPrefillUnsupported, got %v", converse, i, err)
			}
		}
	}
}
