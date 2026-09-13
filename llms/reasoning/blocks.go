package reasoning

import "strings"

// Block is one reasoning block as the vendor produced it: readable text with the
// signature that covers it, or encrypted data, never both.
type Block struct {
	Text      string `json:"text,omitempty"`
	Signature []byte `json:"signature,omitempty"`
	Redacted  []byte `json:"redacted,omitempty"`

	// AfterToolCalls counts the tool calls the vendor put ahead of this block in
	// the same response.
	AfterToolCalls int `json:"after_tool_calls,omitempty"`
}

// FromBlocks builds the reasoning of one response. A lone plain block comes
// back as Content and Signature with no Blocks, so read blocks through Sequence.
func FromBlocks(blocks []Block) *ContentReasoning {
	switch {
	case len(blocks) == 0:
		return nil
	case len(blocks) == 1 && blocks[0].Redacted == nil && blocks[0].AfterToolCalls == 0:
		return &ContentReasoning{Content: blocks[0].Text, Signature: blocks[0].Signature}
	}

	var text strings.Builder
	for _, block := range blocks {
		text.WriteString(block.Text)
	}
	return &ContentReasoning{Content: text.String(), Blocks: blocks}
}

func (r *ContentReasoning) Sequence() []Block {
	if r == nil {
		return nil
	}
	if len(r.Blocks) > 0 {
		return r.Blocks
	}

	if r.Content == "" && len(r.Signature) == 0 {
		return nil
	}
	return []Block{{Text: r.Content, Signature: r.Signature}}
}

// GroupByToolCalls splits blocks by where they sit in a turn with toolCalls
// tool calls: group 0 opens the turn and group i follows tool call i-1. A block
// placed past the turn's last tool call closes it.
func GroupByToolCalls(blocks []Block, toolCalls int) [][]Block {
	groups := make([][]Block, toolCalls+1)
	for _, block := range blocks {
		at := min(max(block.AfterToolCalls, 0), toolCalls)
		groups[at] = append(groups[at], block)
	}
	return groups
}

// Collector gathers the reasoning blocks of one response in the vendor's order.
type Collector struct {
	blocks    []Block
	toolCalls int
}

func (c *Collector) Thought(text string, signature []byte) {
	if text == "" && len(signature) == 0 {
		return
	}
	c.blocks = append(c.blocks, Block{Text: text, Signature: signature, AfterToolCalls: c.toolCalls})
}

func (c *Collector) Encrypted(data []byte) {
	if len(data) == 0 {
		return
	}
	c.blocks = append(c.blocks, Block{Redacted: data, AfterToolCalls: c.toolCalls})
}

func (c *Collector) ToolCall() {
	c.toolCalls++
}

func (c *Collector) Reasoning() *ContentReasoning {
	return FromBlocks(c.blocks)
}
