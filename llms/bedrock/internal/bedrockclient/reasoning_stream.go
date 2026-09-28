package bedrockclient

import (
	"bytes"
	"maps"
	"slices"
	"strings"

	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type streamedBlock struct {
	toolCall  bool
	text      strings.Builder
	signature bytes.Buffer
	redacted  []byte
}

type streamedReasoning struct {
	blocks map[int32]*streamedBlock
}

func (s *streamedReasoning) at(index int32) *streamedBlock {
	if s.blocks == nil {
		s.blocks = make(map[int32]*streamedBlock)
	}
	block, ok := s.blocks[index]
	if !ok {
		block = &streamedBlock{}
		s.blocks[index] = block
	}
	return block
}

func (s *streamedReasoning) text(index int32, text string) {
	s.at(index).text.WriteString(text)
}

func (s *streamedReasoning) signature(index int32, signature string) {
	s.at(index).signature.WriteString(signature)
}

func (s *streamedReasoning) encrypted(index int32, data []byte) {
	block := s.at(index)
	block.redacted = append(block.redacted, data...)
}

func (s *streamedReasoning) toolCall(index int32) {
	s.at(index).toolCall = true
}

func (s *streamedReasoning) result() *reasoning.ContentReasoning {
	var thoughts reasoning.Collector
	for _, index := range slices.Sorted(maps.Keys(s.blocks)) {
		block := s.blocks[index]
		switch {
		case block.redacted != nil:
			thoughts.Encrypted(block.redacted)
		case block.text.Len() > 0 || block.signature.Len() > 0:
			thoughts.Thought(block.text.String(), bytes.Clone(block.signature.Bytes()))
		case block.toolCall:
			thoughts.ToolCall()
		}
	}
	return thoughts.Reasoning()
}
