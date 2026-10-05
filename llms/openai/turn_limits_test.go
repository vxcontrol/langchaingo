package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func turnLimitErr(t *testing.T, model string, msgs []llms.MessageContent, opts ...llms.CallOption) error {
	t.Helper()

	const completion = `{"id":"x","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completion)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithBaseURL(srv.URL), WithToken("test"), WithModel(model))
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	_, err = llm.GenerateContent(context.Background(), msgs, opts...)
	return err
}

func askedFor(text string) []llms.MessageContent { //nolint:unparam // every caller asks the same thing today
	return []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, text)}
}

func endingOnAssistant() []llms.MessageContent {
	return []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "finish this"),
		llms.TextParts(llms.ChatMessageTypeAI, "the answer is"),
	}
}

var turnLimitTools = llms.WithTools([]llms.Tool{{
	Type:     "function",
	Function: &llms.FunctionDefinition{Name: "calc", Parameters: map[string]any{"type": "object"}},
}})

func TestASystemMessageAfterTheAnswerStaysInPlaceOnTheOpenAITransport(t *testing.T) {
	t.Parallel()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL("http://litellm.internal/v1"), WithModel("anthropic/claude-sonnet-4-6"), WithHTTPClient(doer))
	_, err := llm.GenerateContent(context.Background(),
		append(endingOnAssistant(), llms.TextParts(llms.ChatMessageTypeSystem, "be brief")))
	if err != nil {
		t.Fatalf("the gateway decides what a system message after the answer means: %v", err)
	}
	var body struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(doer.body, &body); err != nil {
		t.Fatal(err)
	}
	roles := make([]string, 0, len(body.Messages))
	for _, m := range body.Messages {
		roles = append(roles, m.Role)
	}
	if !slices.Equal(roles, []string{"user", "assistant", "system"}) {
		t.Errorf("roles on the wire = %v", roles)
	}
}

func TestThePrefillIsJudgedOnTheMessagesTheOpenAITransportSends(t *testing.T) {
	t.Parallel()

	human := llms.TextParts(llms.ChatMessageTypeHuman, "finish this")
	var target *reasoning.ErrAssistantPrefillUnsupported
	for i, messages := range [][]llms.MessageContent{
		{human, llms.TextParts(llms.ChatMessageTypeAI, "")},
		{human, {Role: llms.ChatMessageTypeAI}},
		{human, {Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextContent{Reasoning: &reasoning.ContentReasoning{Content: "hm"}}}}},
	} {
		if err := turnLimitErr(t, "claude-sonnet-4-6", messages); errors.As(err, &target) {
			t.Errorf("conversation %d sends no assistant message last: %v", i, err)
		}
	}
	endsOnTheAnswer := append(endingOnAssistant(), llms.TextParts(llms.ChatMessageTypeHuman, ""))
	if err := turnLimitErr(t, "claude-sonnet-4-6", endsOnTheAnswer); !errors.As(err, &target) {
		t.Errorf("an empty human message is not sent, so the answer goes last: got %v", err)
	}
}

func TestABudgetWithAForcedChoiceAndNoToolsIsNotRefusedOnTheOpenAITransport(t *testing.T) {
	t.Parallel()

	err := turnLimitErr(t, "claude-sonnet-4-5", askedFor("hi"),
		llms.WithReasoning(llms.ReasoningLow, 0), llms.WithToolChoice(llms.ToolChoice{Type: "any"}))
	if err != nil {
		t.Errorf("with no tools the choice never reaches the wire, got %v", err)
	}
}

func TestClaudeTurnLimitsOnTheOpenAITransport(t *testing.T) {
	t.Parallel()

	forced := llms.WithToolChoice(llms.ToolChoice{Type: "any"})
	thinking := llms.WithReasoning(llms.ReasoningLow, 0)
	tools := turnLimitTools

	t.Run("budget thinking with a forced tool is refused", func(t *testing.T) {
		t.Parallel()
		err := turnLimitErr(t, "claude-sonnet-4-5", askedFor("hi"), thinking, tools, forced)
		var target *reasoning.ErrForcedToolUseWithThinking
		if !errors.As(err, &target) {
			t.Errorf("want ErrForcedToolUseWithThinking, got %v", err)
		}
	})

	t.Run("a generation that also thinks adaptively is not refused", func(t *testing.T) {
		t.Parallel()
		if err := turnLimitErr(t, "claude-opus-4-6", askedFor("hi"), thinking, tools, forced); err != nil {
			t.Errorf("this generation answers an effort adaptively, so a forced tool is fine, got %v", err)
		}
	})

	t.Run("a budget sent as the thinking object is refused with a forced tool", func(t *testing.T) {
		t.Parallel()
		err := turnLimitErr(t, "claude-opus-4-6", askedFor("hi"),
			llms.WithReasoning(llms.ReasoningNone, 2048), tools, forced)
		var target *reasoning.ErrForcedToolUseWithThinking
		if !errors.As(err, &target) {
			t.Errorf("want ErrForcedToolUseWithThinking, got %v", err)
		}
	})

	t.Run("an effort with no budget is not refused", func(t *testing.T) {
		t.Parallel()
		if err := turnLimitErr(t, "claude-sonnet-4-5", askedFor("hi"),
			llms.WithReasoning(llms.ReasoningEffort("minimal"), 0), tools, forced); err != nil {
			t.Errorf("an effort the budget mapper rejects sends no thinking, so nothing is refused, got %v", err)
		}
	})

	t.Run("the OpenAI spelling of a forced tool is refused too", func(t *testing.T) {
		t.Parallel()
		for _, choice := range []any{
			"required",
			llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "calc"}},
			map[string]any{"type": "function", "function": map[string]any{"name": "calc"}},
		} {
			err := turnLimitErr(t, "claude-sonnet-4-5", askedFor("hi"),
				thinking, tools, llms.WithToolChoice(choice))
			var target *reasoning.ErrForcedToolUseWithThinking
			if !errors.As(err, &target) {
				t.Errorf("%#v demands a tool, want ErrForcedToolUseWithThinking, got %v", choice, err)
			}
		}
	})

	t.Run("adaptive thinking carries no such limit", func(t *testing.T) {
		t.Parallel()
		if err := turnLimitErr(t, "claude-sonnet-5", askedFor("hi"), thinking, tools, forced); err != nil {
			t.Errorf("an adaptive generation takes a forced tool alongside thinking, got %v", err)
		}
	})

	t.Run("a non-Claude model is not refused", func(t *testing.T) {
		t.Parallel()
		if err := turnLimitErr(t, "gpt-5.2", askedFor("hi"), thinking, tools, forced); err != nil {
			t.Errorf("the rule is Anthropic's, got %v", err)
		}
	})

	t.Run("a generation that rejects prefill is refused", func(t *testing.T) {
		t.Parallel()
		for _, model := range []string{
			"claude-opus-4-6", "anthropic/claude-opus-4.6:nitro", "anthropic/claude-sonnet-4.6:online",
			"anthropic/claude-opus-5:floor",
		} {
			err := turnLimitErr(t, model, endingOnAssistant())
			var target *reasoning.ErrAssistantPrefillUnsupported
			if !errors.As(err, &target) {
				t.Errorf("%s: want ErrAssistantPrefillUnsupported, got %v", model, err)
			}
		}
	})

	t.Run("an older generation still takes a prefill", func(t *testing.T) {
		t.Parallel()
		if err := turnLimitErr(t, "claude-sonnet-4-5", endingOnAssistant()); err != nil {
			t.Errorf("this generation answers a prefilled turn, got %v", err)
		}
	})
}
