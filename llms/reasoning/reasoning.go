package reasoning

import (
	"regexp"
	"strings"
	"sync"
)

const trimCharset = "\t\r\n "

var (
	reMatchStartReasoningContent = regexp.MustCompile(`(?s)^(.*?)<(think|thinking)>(.*?)$`)
	reMatchEndReasoningContent   = regexp.MustCompile(`(?s)^(.*?)</(think|thinking)>(.*?)$`)
	reMatchReasoningContent      = regexp.MustCompile(`(?s)^(.*?)<(think|thinking)>(.*?)</(?:think|thinking)>\s*(.*?)$`)
)

type ChunkContentSplitterState int

const (
	ChunkContentSplitterStateText ChunkContentSplitterState = iota
	ChunkContentSplitterStateReasoning
)

type ChunkContentSplitter interface {
	Split(chunk string) (string, string)
	GetState() ChunkContentSplitterState
}

type chunkContentSplitter struct {
	mx    sync.Mutex
	state ChunkContentSplitterState
}

func NewChunkContentSplitter() ChunkContentSplitter {
	return &chunkContentSplitter{}
}

func (c *chunkContentSplitter) Split(chunk string) (string, string) {
	if c == nil { // splitter is not initialized and it does not work
		return "", chunk
	}

	c.mx.Lock()
	defer c.mx.Unlock()

	if c.state == ChunkContentSplitterStateReasoning {
		matches := reMatchEndReasoningContent.FindStringSubmatch(chunk)
		if len(matches) < 3 {
			return "", chunk
		}

		c.state = ChunkContentSplitterStateText
		suffix := matches[1]
		chunk = matches[3]
		return chunk, suffix
	}

	matches := reMatchStartReasoningContent.FindStringSubmatch(chunk)
	if len(matches) < 3 {
		return chunk, ""
	}

	c.state = ChunkContentSplitterStateReasoning
	prefix := matches[1]
	reasoning := matches[3]

	matches = reMatchEndReasoningContent.FindStringSubmatch(reasoning)
	if len(matches) < 3 {
		return prefix, reasoning
	}

	c.state = ChunkContentSplitterStateText
	suffix := matches[1]
	chunk = matches[3]
	return prefix + " " + chunk, suffix
}

func (c *chunkContentSplitter) GetState() ChunkContentSplitterState {
	if c == nil { // splitter is not initialized and returns default state
		return ChunkContentSplitterStateText
	}

	c.mx.Lock()
	defer c.mx.Unlock()

	return c.state
}

func SplitContent(content string) (string, string) {
	content = strings.Trim(content, trimCharset)

	matches := reMatchReasoningContent.FindStringSubmatch(content)
	if len(matches) < 5 {
		return "", content
	}

	prefix := strings.Trim(matches[1], trimCharset)
	reasoning := strings.Trim(matches[3], trimCharset)
	text := strings.Trim(matches[4], trimCharset)

	if prefix != "" {
		text = prefix + "\n" + text
	}

	return reasoning, text
}
