// Package answer keeps the blocks of a model's answer in the order the vendor returned them.
package answer

import (
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type Parts []llms.ContentPart

func (p *Parts) Text(text string) {
	*p = append(*p, llms.TextContent{Text: text})
}

func (p *Parts) ToolCall(call llms.ToolCall) {
	*p = append(*p, call)
}

func (p Parts) With(r *reasoning.ContentReasoning) []llms.ContentPart {
	if len(p) == 0 || r.IsEmpty() {
		return p
	}
	if text, ok := p[0].(llms.TextContent); ok {
		text.Reasoning = r
		return append([]llms.ContentPart{text}, p[1:]...)
	}
	return append([]llms.ContentPart{llms.TextContent{Reasoning: r}}, p...)
}
