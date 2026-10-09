package openaiclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/vxcontrol/langchaingo/internal/streamend"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type ResponsesRequest struct {
	Model             string              `json:"model"`
	Input             []any               `json:"input"`
	MaxOutputTokens   *int                `json:"max_output_tokens,omitempty"`
	Temperature       *float64            `json:"temperature,omitempty"`
	TopP              *float64            `json:"top_p,omitempty"`
	Reasoning         *ResponsesReasoning `json:"reasoning,omitempty"`
	Tools             []ResponsesTool     `json:"tools,omitempty"`
	ToolChoice        any                 `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool               `json:"parallel_tool_calls,omitempty"`
	Text              *ResponsesText      `json:"text,omitempty"`
	Store             bool                `json:"store"`
	Stream            bool                `json:"stream,omitempty"`
	Metadata          map[string]any      `json:"metadata,omitempty"`

	ExtraBody     map[string]any     `json:"-"`
	StreamingFunc streaming.Callback `json:"-"`
}

type ResponsesReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type ResponsesTool struct {
	Type        string `json:"type"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters"`
	Strict      bool   `json:"strict"`
}

type ResponsesText struct {
	Format    *ResponsesFormat `json:"format,omitempty"`
	Verbosity string           `json:"verbosity,omitempty"`
}

type ResponsesFormat struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

type ResponsesMessage struct {
	Type    string `json:"type"`
	Role    string `json:"role"`
	Content any    `json:"content"`
	Phase   string `json:"phase,omitempty"`
}

type ResponsesInputPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

type ResponsesFunctionCall struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ResponsesFunctionCallOutput struct {
	Type   string `json:"type"`
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

type ResponsesReasoningItem struct {
	Type             string             `json:"type"`
	ID               string             `json:"id"`
	Summary          []ResponsesSummary `json:"summary"`
	EncryptedContent string             `json:"encrypted_content,omitempty"`
}

type ResponsesSummary struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ResponsesResponse struct {
	ID                string                `json:"id"`
	Model             string                `json:"model"`
	Status            string                `json:"status"`
	Error             *providerError        `json:"error"`
	IncompleteDetails *ResponsesIncomplete  `json:"incomplete_details"`
	Output            []ResponsesOutputItem `json:"output"`
	Usage             *ResponsesUsage       `json:"usage"`
}

type ResponsesIncomplete struct {
	Reason string `json:"reason"`
}

type ResponsesOutputItem struct {
	Type             string                   `json:"type"`
	ID               string                   `json:"id"`
	Status           string                   `json:"status,omitempty"`
	Phase            string                   `json:"phase,omitempty"`
	Content          []ResponsesOutputContent `json:"content,omitempty"`
	Summary          []ResponsesSummary       `json:"summary,omitempty"`
	EncryptedContent string                   `json:"encrypted_content,omitempty"`
	CallID           string                   `json:"call_id,omitempty"`
	Name             string                   `json:"name,omitempty"`
	Arguments        string                   `json:"arguments,omitempty"`
}

type ResponsesOutputContent struct {
	Type    string `json:"type"`
	Text    string `json:"text,omitempty"`
	Refusal string `json:"refusal,omitempty"`
}

type ResponsesUsage struct {
	InputTokens        int `json:"input_tokens"`
	InputTokensDetails struct {
		CachedTokens     int `json:"cached_tokens"`
		CacheWriteTokens int `json:"cache_write_tokens"`
	} `json:"input_tokens_details"`
	OutputTokens        int `json:"output_tokens"`
	OutputTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
	TotalTokens int `json:"total_tokens"`
}

func (r *ResponsesResponse) providerError() error {
	if r.Status == "failed" && r.Error != nil {
		return r.Error.asError()
	}
	return nil
}

func (c *Client) CreateResponse(ctx context.Context, payload *ResponsesRequest) (*ResponsesResponse, error) {
	payload.Stream = payload.StreamingFunc != nil
	metadata := payload.Metadata
	payload.Metadata = withoutInternalMetadata(metadata)
	payloadBytes, err := json.Marshal(payload)
	payload.Metadata = metadata
	if err != nil {
		return nil, err
	}
	if payloadBytes, err = mergeExtraBody(payloadBytes, payload.ExtraBody); err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.buildURL("/responses", payload.Model), bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, err
	}
	c.setHeaders(req)
	r, err := c.httpClient.Do(req)
	if err != nil {
		return nil, sanitizeHTTPError(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		return nil, statusError(r.StatusCode, r.Body)
	}

	if payload.Stream {
		return parseResponsesStream(ctx, r.Body, payload.StreamingFunc)
	}
	var response ResponsesResponse
	if err := json.NewDecoder(r.Body).Decode(&response); err != nil {
		return nil, err
	}
	return &response, response.providerError()
}

func withoutInternalMetadata(metadata map[string]any) map[string]any {
	kept := make(map[string]any, len(metadata))
	for key, value := range metadata {
		if !strings.HasPrefix(key, "openai:") {
			kept[key] = value
		}
	}
	if len(kept) == 0 {
		return nil
	}
	return kept
}

type responsesEvent struct {
	Type     string              `json:"type"`
	Delta    string              `json:"delta"`
	Item     ResponsesOutputItem `json:"item"`
	Response *ResponsesResponse  `json:"response"`
	Code     any                 `json:"code"`
	Message  string              `json:"message"`
}

func parseResponsesStream(ctx context.Context, body io.Reader, callback streaming.Callback) (*ResponsesResponse, error) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, initialStreamBuffer), maxStreamLine)
	var final *ResponsesResponse
	for final == nil && scanner.Scan() {
		data, isData := strings.CutPrefix(scanner.Text(), "data:")
		data = strings.TrimSpace(data)
		if !isData || !looksLikeJSONObject(data) {
			continue
		}
		var event responsesEvent
		if json.Unmarshal([]byte(data), &event) != nil {
			continue
		}
		var err error
		switch event.Type {
		case "response.output_text.delta":
			err = streaming.CallWithText(ctx, callback, event.Delta)
		case "response.reasoning_summary_text.delta":
			err = streaming.CallWithReasoningContent(ctx, callback, event.Delta)
		case "response.output_item.done":
			if event.Item.Type == "function_call" {
				err = streaming.CallWithToolCall(ctx, callback,
					streaming.NewToolCall(event.Item.CallID, event.Item.Name, event.Item.Arguments))
			}
		case "response.completed", "response.incomplete", "response.failed":
			final = event.Response
		case "error":
			return nil, (&providerError{Message: event.Message, Code: event.Code}).asError()
		}
		if err != nil {
			return nil, fmt.Errorf("streaming func returned an error: %w", err)
		}
	}
	if final == nil {
		return nil, streamend.Incomplete(ctx, scanner.Err())
	}
	return final, final.providerError()
}

func (r *ChatRequest) ResponsesFormat() *ResponsesFormat {
	switch f := r.ResponseFormat; {
	case f == nil:
		return nil
	case f.raw != nil:
		return &ResponsesFormat{
			Type: "json_schema", Name: f.raw.Name, Description: f.raw.Description, Schema: f.raw.Schema, Strict: &f.raw.Strict,
		}
	case f.typed == nil || f.typed.Type == "text":
		return nil
	case f.typed.JSONSchema != nil:
		schema, err := json.Marshal(f.typed.JSONSchema.Schema)
		if err != nil {
			return nil
		}
		return &ResponsesFormat{Type: "json_schema", Name: f.typed.JSONSchema.Name, Schema: schema, Strict: &f.typed.JSONSchema.Strict}
	}
	return &ResponsesFormat{Type: r.ResponseFormat.typed.Type}
}

func (r *ResponsesResponse) ChatResponse() *ChatCompletionResponse {
	var text, refusal strings.Builder
	var calls []ToolCall
	var thoughts reasoning.Collector
	for _, item := range r.Output {
		switch item.Type {
		case "reasoning":
			var summary strings.Builder
			for _, part := range item.Summary {
				summary.WriteString(part.Text)
			}
			thoughts.Item(item.ID, summary.String(), []byte(item.EncryptedContent))
		case "message":
			for _, part := range item.Content {
				text.WriteString(part.Text)
				refusal.WriteString(part.Refusal)
			}
		case "function_call":
			calls = append(calls, ToolCall{ID: item.CallID, Type: ToolTypeFunction, Function: ToolFunction{Name: item.Name, Arguments: item.Arguments}})
			thoughts.ToolCall()
		}
	}

	finish := FinishReasonStop
	switch {
	case r.IncompleteDetails != nil && r.IncompleteDetails.Reason == "max_output_tokens":
		finish = FinishReasonLength
	case r.IncompleteDetails != nil && r.IncompleteDetails.Reason == "content_filter":
		finish = FinishReasonContentFilter
	case len(calls) > 0:
		finish = FinishReasonToolCalls
	}
	chat := &ChatCompletionResponse{ID: r.ID, Model: r.Model, Choices: []*ChatCompletionChoice{{
		Message:      ChatMessage{Role: "assistant", Content: text.String(), Refusal: refusal.String(), ToolCalls: calls},
		FinishReason: finish,
		Reasoning:    thoughts.Reasoning(),
	}}}
	if usage := r.Usage; usage != nil {
		chat.Usage.PromptTokens = usage.InputTokens
		chat.Usage.CompletionTokens = usage.OutputTokens
		chat.Usage.TotalTokens = usage.TotalTokens
		chat.Usage.PromptTokensDetails.CachedTokens = usage.InputTokensDetails.CachedTokens
		chat.Usage.PromptTokensDetails.CacheWriteTokens = usage.InputTokensDetails.CacheWriteTokens
		chat.Usage.CompletionTokensDetails.ReasoningTokens = usage.OutputTokensDetails.ReasoningTokens
	}
	return chat
}
