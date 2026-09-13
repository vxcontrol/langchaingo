package bedrockclient

import (
	"bytes"
	"slices"
	"strings"

	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type streamedThought struct {
	text      strings.Builder
	signature bytes.Buffer
	redacted  []byte
}

type streamedReasoning struct {
	thoughts  map[int32]*streamedThought
	toolCalls map[int32]bool
}

func (s *streamedReasoning) at(index int32) *streamedThought {
	if s.thoughts == nil {
		s.thoughts = make(map[int32]*streamedThought)
	}
	thought, ok := s.thoughts[index]
	if !ok {
		thought = &streamedThought{}
		s.thoughts[index] = thought
	}
	return thought
}

func (s *streamedReasoning) toolCall(index int32) {
	if s.toolCalls == nil {
		s.toolCalls = make(map[int32]bool)
	}
	s.toolCalls[index] = true
}

func (s *streamedReasoning) result() *reasoning.ContentReasoning {
	indexes := make([]int32, 0, len(s.thoughts)+len(s.toolCalls))
	for index := range s.thoughts {
		indexes = append(indexes, index)
	}
	for index := range s.toolCalls {
		if _, isThought := s.thoughts[index]; !isThought {
			indexes = append(indexes, index)
		}
	}
	slices.Sort(indexes)

	var thoughts reasoning.Collector
	for _, index := range indexes {
		thought, isThought := s.thoughts[index]
		switch {
		case !isThought:
			thoughts.ToolCall()
		case thought.redacted != nil:
			thoughts.Encrypted(thought.redacted)
		default:
			thoughts.Thought(thought.text.String(), bytes.Clone(thought.signature.Bytes()))
		}
	}
	return thoughts.Reasoning()
}
