package reasoning

import (
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChunkContentSplitter_Creation(t *testing.T) {
	t.Parallel()

	splitter := NewChunkContentSplitter()
	require.NotNil(t, splitter)

	// Test initial state
	assert.Equal(t, ChunkContentSplitterStateText, splitter.GetState())

	// Test nil splitter
	var nilSplitter *chunkContentSplitter
	assert.Equal(t, ChunkContentSplitterStateText, nilSplitter.GetState())

	// Create multiple splitters and check they're independent
	splitter1 := NewChunkContentSplitter()
	splitter2 := NewChunkContentSplitter()

	assert.Equal(t, ChunkContentSplitterStateText, splitter1.GetState())
	assert.Equal(t, ChunkContentSplitterStateText, splitter2.GetState())

	// Change state of one splitter
	text, _ := splitter1.Split("Start <thinking>reasoning")
	assert.Equal(t, "Start ", text)
	assert.Equal(t, ChunkContentSplitterStateReasoning, splitter1.GetState())

	// Second splitter should remain unchanged
	assert.Equal(t, ChunkContentSplitterStateText, splitter2.GetState())
}

func TestChunkContentSplitter_SimpleTests(t *testing.T) {
	t.Parallel()

	// Simple text
	splitter := NewChunkContentSplitter()
	text, reasoning := splitter.Split("Hello, world!")
	assert.Equal(t, "Hello, world!", text)
	assert.Equal(t, "", reasoning)
	assert.Equal(t, ChunkContentSplitterStateText, splitter.GetState())

	// Text with thinking tag
	splitter = NewChunkContentSplitter()
	text, reasoning = splitter.Split("Hello <thinking>This is a reasoning step</thinking> world!")
	assert.Equal(t, "Hello   world!", text)
	assert.Equal(t, "This is a reasoning step", reasoning)
	assert.Equal(t, ChunkContentSplitterStateText, splitter.GetState())

	// Only start thinking tag
	splitter = NewChunkContentSplitter()
	text, reasoning = splitter.Split("Hello <thinking>This is a reasoning step")
	assert.Equal(t, "Hello ", text)
	assert.Equal(t, "This is a reasoning step", reasoning)
	assert.Equal(t, ChunkContentSplitterStateReasoning, splitter.GetState())

	// Only end thinking tag
	splitter = NewChunkContentSplitter()
	text, reasoning = splitter.Split("This is a reasoning step</thinking> Hello")
	assert.Equal(t, "This is a reasoning step</thinking> Hello", text)
	assert.Equal(t, "", reasoning)
	assert.Equal(t, ChunkContentSplitterStateText, splitter.GetState())

	// Empty string
	splitter = NewChunkContentSplitter()
	text, reasoning = splitter.Split("")
	assert.Equal(t, "", text)
	assert.Equal(t, "", reasoning)
	assert.Equal(t, ChunkContentSplitterStateText, splitter.GetState())
}

func TestBasicMultipleChunks(t *testing.T) {
	t.Parallel()

	// Test basic reasoning across chunks
	splitter := NewChunkContentSplitter()

	// First chunk - start thinking
	text1, reason1 := splitter.Split("Hello <thinking>This is")
	assert.Equal(t, "Hello ", text1)
	assert.Equal(t, "This is", reason1)
	assert.Equal(t, ChunkContentSplitterStateReasoning, splitter.GetState())

	// Second chunk - continue reasoning
	text2, reason2 := splitter.Split(" reasoning across chunks")
	assert.Equal(t, "", text2)
	assert.Equal(t, " reasoning across chunks", reason2)
	assert.Equal(t, ChunkContentSplitterStateReasoning, splitter.GetState())

	// Third chunk - end reasoning
	text3, reason3 := splitter.Split("</thinking> world!")
	assert.Equal(t, " world!", text3)
	assert.Equal(t, "", reason3)
	assert.Equal(t, ChunkContentSplitterStateText, splitter.GetState())

	// Test multiple reasoning segments in same splitter
	splitter = NewChunkContentSplitter()

	// First segment
	text1, reason1 = splitter.Split("Start <thinking>First</thinking> middle")
	assert.Equal(t, "Start   middle", text1)
	assert.Equal(t, "First", reason1)
	assert.Equal(t, ChunkContentSplitterStateText, splitter.GetState())
}

//nolint:funlen
func TestSplitContent(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name           string
		input          string
		expectedReason string
		expectedText   string
	}{
		{
			name:           "simple text with reasoning",
			input:          "Hello <thinking>This is reasoning</thinking> world!",
			expectedReason: "This is reasoning",
			expectedText:   "Hello\nworld!",
		},
		{
			name:           "text with think tag",
			input:          "Let me explain: <think>First analyze the data\nThen draw conclusions</think> The answer is 42.",
			expectedReason: "First analyze the data\nThen draw conclusions",
			expectedText:   "Let me explain:\nThe answer is 42.",
		},
		{
			name:           "text with prefix and reasoning",
			input:          "Before analysis <thinking>My thought process</thinking> After analysis",
			expectedReason: "My thought process",
			expectedText:   "Before analysis\nAfter analysis",
		},
		{
			name:           "no reasoning tags",
			input:          "Plain text without reasoning",
			expectedReason: "",
			expectedText:   "Plain text without reasoning",
		},
		{
			name:           "multiple reasoning blocks - only first is extracted",
			input:          "Start <thinking>First reasoning</thinking> middle <thinking>Second reasoning</thinking> end",
			expectedReason: "First reasoning",
			expectedText:   "Start\nmiddle <thinking>Second reasoning</thinking> end",
		},
		{
			name:           "trimming whitespace",
			input:          "  Start  <thinking>  Reasoning with spaces  </thinking>  End  ",
			expectedReason: "Reasoning with spaces",
			expectedText:   "Start\nEnd",
		},
		{
			name:           "empty reasoning content",
			input:          "Text <thinking></thinking> more text",
			expectedReason: "",
			expectedText:   "Text\nmore text",
		},
		{
			name:           "empty string",
			input:          "",
			expectedReason: "",
			expectedText:   "",
		},
		{
			name:           "mixed think and thinking tags",
			input:          "Start <think>First <thinking>nested</thinking> second</think> end",
			expectedReason: "First <thinking>nested",
			expectedText:   "Start\nsecond</think> end",
		},
		{
			name:           "malformed reasoning tag",
			input:          "Start <thinkng>This is not a valid tag</thinkng> end",
			expectedReason: "",
			expectedText:   "Start <thinkng>This is not a valid tag</thinkng> end",
		},
		{
			name:           "reasoning at start",
			input:          "<thinking>First reasoning</thinking> Text after",
			expectedReason: "First reasoning",
			expectedText:   "Text after",
		},
		{
			name:           "reasoning at end",
			input:          "Text before <thinking>Last reasoning</thinking>",
			expectedReason: "Last reasoning",
			expectedText:   "Text before\n",
		},
		{
			name:           "multiple lines with reasoning",
			input:          "Line 1\nLine 2\n<thinking>Reasoning\non multiple\nlines</thinking>\nLine 3\nLine 4",
			expectedReason: "Reasoning\non multiple\nlines",
			expectedText:   "Line 1\nLine 2\nLine 3\nLine 4",
		},
		{
			name:           "incomplete open tag",
			input:          "Text <thinki",
			expectedReason: "",
			expectedText:   "Text <thinki",
		},
		{
			name:           "reasoning tag with attributes",
			input:          "Text <thinking id=\"123\">Reasoning</thinking> more text",
			expectedReason: "",
			expectedText:   "Text <thinking id=\"123\">Reasoning</thinking> more text",
		},
	}

	for idx := range testCases {
		tc := testCases[idx]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			reasoning, text := SplitContent(tc.input)
			assert.Equal(t, tc.expectedReason, reasoning)
			assert.Equal(t, tc.expectedText, text)

			// Additional checks for edge cases
			if tc.input == "" {
				assert.Empty(t, reasoning)
				assert.Empty(t, text)
			}

			if !strings.Contains(tc.input, "thinking") && !strings.Contains(tc.input, "think") {
				assert.Empty(t, reasoning)
				assert.Equal(t, tc.input, text)
			}
		})
	}
}

func TestNilChunkContentSplitter(t *testing.T) {
	t.Parallel()

	var nilSplitter *chunkContentSplitter
	text, reasoning := nilSplitter.Split("Hello <thinking>reasoning</thinking> world")

	// A nil splitter should pass the content through as reasoning
	assert.Equal(t, "", text)
	assert.Equal(t, "Hello <thinking>reasoning</thinking> world", reasoning)

	// Multiple calls to nil splitter shouldn't crash
	text, reasoning = nilSplitter.Split("Another piece of text")
	assert.Equal(t, "", text)
	assert.Equal(t, "Another piece of text", reasoning)

	text, reasoning = nilSplitter.Split("")
	assert.Equal(t, "", text)
	assert.Equal(t, "", reasoning)

	// Nil splitter GetState should return default state
	assert.Equal(t, ChunkContentSplitterStateText, nilSplitter.GetState())
}

func TestChunkContentSplitterEdgeCases(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name               string
		initialState       ChunkContentSplitterState
		input              string
		expectedText       string
		expectedReasoning  string
		expectedFinalState ChunkContentSplitterState
	}{
		{
			name:               "start in reasoning state with no end tag",
			initialState:       ChunkContentSplitterStateReasoning,
			input:              "continuing reasoning with no end tag",
			expectedText:       "",
			expectedReasoning:  "continuing reasoning with no end tag",
			expectedFinalState: ChunkContentSplitterStateReasoning,
		},
		{
			name:               "start in reasoning state with end tag",
			initialState:       ChunkContentSplitterStateReasoning,
			input:              "reasoning</thinking> and now text",
			expectedText:       " and now text",
			expectedReasoning:  "reasoning",
			expectedFinalState: ChunkContentSplitterStateText,
		},
		{
			name:               "start tag inside reasoning state",
			initialState:       ChunkContentSplitterStateReasoning,
			input:              "first part <thinking>second part",
			expectedText:       "",
			expectedReasoning:  "first part <thinking>second part",
			expectedFinalState: ChunkContentSplitterStateReasoning,
		},
		{
			name:               "empty input in reasoning state",
			initialState:       ChunkContentSplitterStateReasoning,
			input:              "",
			expectedText:       "",
			expectedReasoning:  "",
			expectedFinalState: ChunkContentSplitterStateReasoning,
		},
		{
			name:               "multiple end tags in reasoning state",
			initialState:       ChunkContentSplitterStateReasoning,
			input:              "reasoning</thinking> text </thinking> more text",
			expectedText:       " text </thinking> more text",
			expectedReasoning:  "reasoning",
			expectedFinalState: ChunkContentSplitterStateText,
		},
	}

	for idx := range testCases {
		tc := testCases[idx]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			splitter := &chunkContentSplitter{state: tc.initialState}
			text, reasoning := splitter.Split(tc.input)

			assert.Equal(t, tc.expectedText, text)
			assert.Equal(t, tc.expectedReasoning, reasoning)
			assert.Equal(t, tc.expectedFinalState, splitter.GetState())

			// Check state transitions
			if tc.initialState != tc.expectedFinalState {
				// We had a state transition
				if tc.initialState == ChunkContentSplitterStateReasoning &&
					tc.expectedFinalState == ChunkContentSplitterStateText {
					assert.Contains(t, tc.input, "</thinking>", "state transition without end tag")
				}
			}
		})
	}
}

func TestStatefulChunkContentSplitting(t *testing.T) {
	t.Parallel()

	// Test a basic sequence of operations to verify the state transitions
	splitter := NewChunkContentSplitter()

	// Initial state should be Text
	assert.Equal(t, ChunkContentSplitterStateText, splitter.GetState())

	// 1. Send text with start tag -> should switch to Reasoning state
	text, reasoning := splitter.Split("Before <thinking>First part")
	assert.Equal(t, "Before ", text)
	assert.Equal(t, "First part", reasoning)
	assert.Equal(t, ChunkContentSplitterStateReasoning, splitter.GetState())

	// 2. Send more content while in Reasoning state -> should stay in Reasoning
	text, reasoning = splitter.Split(" of reasoning")
	assert.Equal(t, "", text)
	assert.Equal(t, " of reasoning", reasoning)
	assert.Equal(t, ChunkContentSplitterStateReasoning, splitter.GetState())

	// 3. Send end tag -> should switch back to Text state
	text, reasoning = splitter.Split("</thinking> After")
	assert.Equal(t, " After", text)
	assert.Equal(t, "", reasoning)
	assert.Equal(t, ChunkContentSplitterStateText, splitter.GetState())

	// 4. Send text with full tag in one chunk -> should process it and stay in Text
	text, reasoning = splitter.Split("More text <thinking>Quick thought</thinking> and more")
	assert.Equal(t, "More text   and more", text)
	assert.Equal(t, "Quick thought", reasoning)
	assert.Equal(t, ChunkContentSplitterStateText, splitter.GetState())
}

func TestConcurrentChunkContentSplitting(t *testing.T) {
	t.Parallel()

	// Create multiple splitters and use them concurrently
	const numSplitters = 5
	const numChunks = 3

	splitters := make([]ChunkContentSplitter, numSplitters)
	for i := range splitters {
		splitters[i] = NewChunkContentSplitter()
	}

	// Define test chunks - simple case that won't cause issues
	chunks := []string{
		"Start text",
		" with <thinking>reasoning</thinking>",
		" and more text",
	}

	// Run concurrent operations
	var wg sync.WaitGroup
	wg.Add(numSplitters)

	for idx := 0; idx < numSplitters; idx++ {
		go func(i int) {
			defer wg.Done()

			splitter := splitters[i]

			for j := 0; j < numChunks; j++ {
				chunkIndex := j % len(chunks)
				chunk := chunks[chunkIndex]

				text, reasoning := splitter.Split(chunk)

				// We don't need to check results, just ensure no race conditions occur
				_ = text
				_ = reasoning
				_ = splitter.GetState()
			}
		}(idx)
	}

	wg.Wait()

	// Verify final states are correct
	for i, splitter := range splitters {
		state := splitter.GetState()
		assert.Equal(t, ChunkContentSplitterStateText, state, "splitter %d in wrong final state", i)
	}
}

func TestIntegration(t *testing.T) {
	t.Parallel()

	// Test basic integration example
	input := "I'll solve this problem step by step. <thinking>First, I need to analyze the question.\nThe key insights are:\n1. X = 10\n2. Y = 5</thinking> Based on my analysis, X + Y = 15"

	reasoning, text := SplitContent(input)

	assert.Equal(t, "First, I need to analyze the question.\nThe key insights are:\n1. X = 10\n2. Y = 5", reasoning)
	assert.Equal(t, "I'll solve this problem step by step.\nBased on my analysis, X + Y = 15", text)

	// Test with code example - just verify basic content extraction works
	codeExample := `Let me analyze this code. <thinking>
This is a simple function that does X and Y.
Time complexity: O(n)
</thinking>

Based on my analysis, this code is efficient.`

	reasoning, text = SplitContent(codeExample)
	assert.Contains(t, reasoning, "Time complexity: O(n)")
	assert.Contains(t, text, "Let me analyze this code.")
	assert.Contains(t, text, "Based on my analysis, this code is efficient.")
}

// BenchmarkSplitContentLargeString tests the performance of SplitContent on a large text.
func BenchmarkSplitContentLargeString(b *testing.B) {
	var builder strings.Builder
	for i := 0; i < 1000; i++ {
		builder.WriteString(fmt.Sprintf("Text segment %d. ", i))
		if i == 500 {
			builder.WriteString("<thinking>This is a reasoning block within a very large text. It should be extracted effectively even in large documents.</thinking> ") //nolint:lll
		}
	}
	largeText := builder.String()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoning, text := SplitContent(largeText)
		if reasoning == "" || text == "" {
			b.Fatal("Empty result")
		}
	}
}

// BenchmarkChunkContentSplitter measures the performance of processing a stream of chunks.
func BenchmarkChunkContentSplitter(b *testing.B) {
	chunks := []string{
		"Start text with ",
		"<thinking>First reasoning step: ",
		"analyze the problem carefully",
		" and identify key components",
		"</thinking> After reasoning, ",
		"<thinking>Second reasoning step: ",
		"solve each part systematically",
		"</thinking> Final conclusion.",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitter := NewChunkContentSplitter()
		var textResult, reasoningResult strings.Builder

		for _, chunk := range chunks {
			text, reasoning := splitter.Split(chunk)
			textResult.WriteString(text)
			reasoningResult.WriteString(reasoning)
		}

		if textResult.Len() == 0 || reasoningResult.Len() == 0 {
			b.Fatal("Empty result")
		}
	}
}
