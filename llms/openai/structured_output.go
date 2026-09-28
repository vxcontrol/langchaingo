package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai/internal/openaiclient"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/structuredoutput"
)

const providerOpenAI = "openai"

const takesNoResponseFormat = "the vendor's chat completions API takes no response_format for this model"

// ErrStructuredOutputRefusal reports that the model declined a structured-output
// request (OpenAI Structured Outputs). A refusal may legitimately not match the
// schema, so it is a distinct typed outcome rather than a validation failure. The
// usage-carrying ContentResponse is returned alongside this error.
type ErrStructuredOutputRefusal struct {
	Model   string
	Choice  int
	Refusal string

	cause error
}

func (e *ErrStructuredOutputRefusal) Unwrap() error { return e.cause }

func (e *ErrStructuredOutputRefusal) Error() string {
	return fmt.Sprintf("openai structured output: model refused (model=%s choice=%d): %s", e.Model, e.Choice, e.Refusal)
}

func noJSONObjectReason(model string) string {
	switch {
	case reasoning.TakesNoResponseFormat(model):
		return takesNoResponseFormat
	case reasoning.TakesNoJSONObject(model):
		return "the vendor has no json_object response format for this model"
	}
	return ""
}

func setJSONMode(req *openaiclient.ChatRequest, model string, opts llms.CallOptions, warn *llms.Warnings) {
	if !opts.GetJSONMode() {
		return
	}
	reason := noJSONObjectReason(model)
	if reason == "" {
		req.SetResponseFormat(ResponseFormatJSON)
		return
	}
	if opts.StructuredOutput == nil {
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithJSONMode", Model: model,
			Asked: "true", Reason: reason,
		})
	}
}

func setClientResponseFormat(req *openaiclient.ChatRequest, model string, rf *ResponseFormat, warn *llms.Warnings) {
	if rf == nil {
		return
	}
	if rf.Type == ResponseFormatJSON.Type {
		if reason := noJSONObjectReason(model); reason != "" {
			warn.Add(llms.Warning{
				Kind: llms.WarningDrop, Option: "WithResponseFormat", Model: model,
				Asked: rf.Type, Reason: reason,
			})
			return
		}
	}
	req.SetResponseFormat(rf)
}

// setStructuredOutput translates a per-call llms.StructuredOutput into OpenAI's
// json_schema response format with strict:true. It takes precedence over the
// schema-less JSONMode json_object and returns a typed conflict against a
// client-level response format rather than silently overwriting one. Under
// WithStructuredOutputFallback a model without json_schema gets the schema in
// the prompt instead, with json_object where its vendor has it, after the same
// checks.
func (o *LLM) setStructuredOutput(req *openaiclient.ChatRequest, opts llms.CallOptions, warn *llms.Warnings) error {
	so := opts.StructuredOutput
	if so == nil {
		return nil
	}
	if err := opts.ValidateStructuredOutput(); err != nil {
		return err
	}
	if so.Name == "" {
		return fmt.Errorf("%w: openai structured output requires a schema name", llms.ErrStructuredOutputConfig)
	}
	if o.client.ResponseFormat != nil {
		return &llms.ErrStructuredOutputConflict{
			Provider: providerOpenAI,
			Detail:   "per-call WithStructuredOutput conflicts with client-level WithResponseFormat",
		}
	}
	model := o.effectiveModel(opts)
	emulated := o.emulatesStructuredOutput(model, opts)
	if reason := openAIStructuredOutputUnsupported(model); reason != "" && !emulated {
		return &llms.ErrStructuredOutputUnsupported{
			Provider: providerOpenAI,
			Model:    model,
			Reason:   reason,
		}
	}
	if err := validateOpenAIStructuredSchema(so.Schema); err != nil {
		return err
	}
	if emulated {
		sent, reason := "a prompt instruction", takesNoResponseFormat
		if noJSONObjectReason(model) == "" {
			req.SetResponseFormat(ResponseFormatJSON)
			sent, reason = "json_object and a prompt instruction", "the vendor's chat completions response_format takes only text and json_object"
		}
		// addToolsToRequest has already moved functions into req.Tools.
		req.Messages = injectSchemaInstruction(req.Messages, so.Schema, len(req.Tools) > 0)
		warn.Add(llms.Warning{
			Kind: llms.WarningSubstitute, Option: "WithStructuredOutput", Model: model,
			Asked: so.Name, Sent: sent,
			Reason: reason + ", so the schema travels in the prompt and the answer is validated locally",
		})
		return nil
	}
	req.SetStructuredOutputSchema(so.Name, so.Description, so.Schema)
	return nil
}

// emulatesStructuredOutput reports whether a structured-output call travels as
// a prompt instruction: the client opted in with WithStructuredOutputFallback,
// and the model's vendor takes json_object but no json_schema, which then goes
// along, or takes no response_format at all.
func (o *LLM) emulatesStructuredOutput(model string, opts llms.CallOptions) bool {
	if opts.StructuredOutput == nil || !o.structuredOutputFallback {
		return false
	}
	return (reasoning.TakesNoJSONSchema(model) && noJSONObjectReason(model) == "") ||
		reasoning.TakesNoResponseFormat(model)
}

// injectSchemaInstruction appends the schema instruction to the last user
// message of the request. The messages are the request's own copies, but their
// part slices are copied again before a part changes. Z.ai answers no
// conversation without a user turn, so when there is none the instruction
// becomes that turn.
func injectSchemaInstruction(msgs []*ChatMessage, schema json.RawMessage, withTools bool) []*ChatMessage {
	instruction := structuredoutput.PromptInstruction(schema, withTools)
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != RoleUser {
			continue
		}
		msg := *msgs[i]
		switch {
		case msg.Content != "":
			msg.Content += "\n\n" + instruction
		default:
			parts := append([]llms.ContentPart(nil), msg.MultiContent...)
			if last := len(parts) - 1; last >= 0 {
				if text, ok := parts[last].(llms.TextContent); ok {
					parts[last] = llms.TextContent{Text: text.Text + "\n\n" + instruction}
					msg.MultiContent = parts
					break
				}
			}
			msg.MultiContent = append(parts, llms.TextContent{Text: instruction})
		}
		msgs[i] = &msg
		return msgs
	}
	return append(msgs, &ChatMessage{Role: RoleUser, MultiContent: []llms.ContentPart{llms.TextContent{Text: instruction}}})
}

// unwrapEmulatedAnswers readies each final answer of an emulated structured
// output call before it is read and validated: the whitespace around it goes, a
// thinking block at its head is taken out, and then the Markdown code fence a
// model put around its whole answer despite the prompt instruction is removed.
// The block becomes the reasoning only when there is none: MiniMax M2.7 and M3
// stream their thinking both as reasoning and inside <think> tags in the
// content, which the client then leaves whole. Only a normal-final choice
// without a tool call is touched.
func unwrapEmulatedAnswers(result *openaiclient.ChatCompletionResponse) {
	for _, choice := range result.Choices {
		if choice == nil || choice.FinishReason != openaiclient.FinishReasonStop || len(choice.Message.ToolCalls) != 0 {
			continue
		}
		msg := &choice.Message
		content := strings.TrimSpace(msg.Content)
		if strings.HasPrefix(content, "<think>") || strings.HasPrefix(content, "<thinking>") {
			var thought string
			thought, content = reasoning.SplitContent(content)
			if msg.ReasoningContent == "" {
				msg.ReasoningContent = thought
			}
		}
		msg.Content = structuredoutput.UnwrapFencedJSON(content)
	}
}

// validateStructuredResponse checks each normal-final ("stop") choice against the
// requested schema. A refusal is surfaced through GenerationInfo, not validated;
// length, content_filter, tool_calls and function_call are not final JSON either.
func (o *LLM) validateStructuredResponse(result *openaiclient.ChatCompletionResponse, opts llms.CallOptions) error {
	so := opts.StructuredOutput
	if so == nil {
		return nil
	}
	model := o.effectiveModel(opts)
	for i, c := range result.Choices {
		if c.FinishReason != openaiclient.FinishReasonStop {
			continue
		}
		if err := structuredoutput.Validate(so.Schema, providerOpenAI, model, i, string(c.FinishReason), c.Message.Content); err != nil {
			return err
		}
	}
	return nil
}

// openAIStructuredOutputUnsupported names why a model KNOWN to lack Structured
// Outputs (json_schema) cannot take one, or returns "". Unknown or newer names
// pass through so the local table never blocks a future model.
func openAIStructuredOutputUnsupported(model string) string {
	const predates = "model predates Structured Outputs (json_schema)"

	m := strings.ToLower(model)
	if idx := strings.LastIndex(m, "/"); idx != -1 {
		m = m[idx+1:]
	}
	switch {
	case reasoning.TakesNoJSONSchema(model):
		return "the vendor's chat completions response_format takes only text and json_object"
	case reasoning.TakesNoResponseFormat(model):
		return takesNoResponseFormat
	case strings.HasPrefix(m, "gpt-3.5"):
		return predates
	case m == "gpt-4", strings.HasPrefix(m, "gpt-4-0"), strings.HasPrefix(m, "gpt-4-32k"), strings.HasPrefix(m, "gpt-4-turbo"):
		return predates
	case m == "gpt-4o-2024-05-13":
		// The first gpt-4o snapshot predates json_schema (added in 2024-08-06).
		return predates
	default:
		return ""
	}
}

// validateOpenAIStructuredSchema enforces the OpenAI Structured Outputs subset that
// is checkable locally: a root object, no top-level anyOf, and every object node
// setting additionalProperties:false with all its properties listed in required.
func validateOpenAIStructuredSchema(raw json.RawMessage) error {
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil {
		return fmt.Errorf("%w: schema must be a JSON object: %w", llms.ErrStructuredOutputConfig, err)
	}
	if _, ok := root["anyOf"]; ok {
		return fmt.Errorf("%w: OpenAI does not allow anyOf at the schema root", llms.ErrStructuredOutputConfig)
	}
	if t, _ := root["type"].(string); t != "object" {
		return fmt.Errorf("%w: OpenAI structured output requires a root object schema", llms.ErrStructuredOutputConfig)
	}
	return checkOpenAIObjectNodes(root)
}

func checkOpenAIObjectNodes(node map[string]any) error {
	props, hasProps := node["properties"].(map[string]any)
	// An object node is any schema whose type is "object" OR that carries
	// properties. OpenAI requires additionalProperties:false on every object,
	// including one with no declared properties (e.g. {"type":"object"}).
	if t, _ := node["type"].(string); t == "object" || hasProps {
		if ap, ok := node["additionalProperties"].(bool); !ok || ap {
			return fmt.Errorf("%w: every object schema must set additionalProperties:false", llms.ErrStructuredOutputConfig)
		}
	}
	if hasProps {
		required := map[string]bool{}
		if reqs, ok := node["required"].([]any); ok {
			for _, r := range reqs {
				if s, ok := r.(string); ok {
					required[s] = true
				}
			}
		}
		for name := range props {
			if !required[name] {
				return fmt.Errorf("%w: property %q must be listed in required (OpenAI strict)", llms.ErrStructuredOutputConfig, name)
			}
		}
	}
	for _, child := range openAISubschemas(node) {
		if m, ok := child.(map[string]any); ok {
			if err := checkOpenAIObjectNodes(m); err != nil {
				return err
			}
		}
	}
	return nil
}

// openAISubschemas returns nested schema nodes worth recursing into for the
// object-node checks (properties, $defs/definitions, items, composition arrays).
func openAISubschemas(node map[string]any) []any {
	var out []any
	if props, ok := node["properties"].(map[string]any); ok {
		for _, v := range props {
			out = append(out, v)
		}
	}
	for _, key := range []string{"$defs", "definitions"} {
		if defs, ok := node[key].(map[string]any); ok {
			for _, v := range defs {
				out = append(out, v)
			}
		}
	}
	if items, ok := node["items"].(map[string]any); ok {
		out = append(out, items)
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if arr, ok := node[key].([]any); ok {
			out = append(out, arr...)
		}
	}
	return out
}
