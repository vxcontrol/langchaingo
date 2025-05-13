package streaming

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolCall(t *testing.T) {
	t.Parallel()

	// Test creating a new tool call
	toolCall := NewToolCall("123", "weather", `{"location": "New York"}`)
	assert.Equal(t, "123", toolCall.ID)
	assert.Equal(t, "weather", toolCall.Name)
	assert.Equal(t, `{"location": "New York"}`, toolCall.Arguments)

	// Test String method
	expected := "ToolCall{ID: 123, Name: weather, Arguments: {\"location\": \"New York\"}}"
	assert.Equal(t, expected, toolCall.String())

	// Test Parse method
	args, err := toolCall.Parse()
	require.NoError(t, err)
	assert.Equal(t, "New York", args["location"])

	// Test Parse with empty arguments
	emptyToolCall := NewToolCall("123", "weather", "")
	args, err = emptyToolCall.Parse()
	require.NoError(t, err)
	assert.Empty(t, args)

	// Test Parse with invalid JSON
	invalidToolCall := NewToolCall("123", "weather", "{invalid json}")
	_, err = invalidToolCall.Parse()
	require.Error(t, err)
}

func TestChunk(t *testing.T) {
	t.Parallel()

	// Test text chunk
	textChunk := NewTextChunk("Hello, world!")
	assert.Equal(t, ChunkTypeText, textChunk.Type)
	assert.Equal(t, "Hello, world!", textChunk.Content)
	assert.Equal(t, "Text: Hello, world!", textChunk.String())

	// Test reasoning chunk
	reasoningChunk := NewReasoningChunk("Step 1: Analyze the problem.")
	assert.Equal(t, ChunkTypeReasoning, reasoningChunk.Type)
	assert.Equal(t, "Step 1: Analyze the problem.", reasoningChunk.ReasoningContent)
	assert.Equal(t, "Reasoning: Step 1: Analyze the problem.", reasoningChunk.String())

	// Test tool call chunk
	toolCall := NewToolCall("123", "weather", `{"location": "New York"}`)
	toolCallChunk := NewToolCallChunk(toolCall)
	assert.Equal(t, ChunkTypeToolCall, toolCallChunk.Type)
	assert.Equal(t, toolCall, toolCallChunk.ToolCall)
	assert.Contains(t, toolCallChunk.String(), "ToolCall: ToolCall{ID: 123")

	// Test done chunk
	doneChunk := NewDoneChunk()
	assert.Equal(t, ChunkTypeDone, doneChunk.Type)
	assert.Equal(t, "Done", doneChunk.String())

	// Test empty chunk type
	noneChunk := Chunk{Type: ChunkTypeNone}
	assert.Equal(t, "None", noneChunk.String())

	// Test Unknown chunk type string representation
	unknownChunk := Chunk{Type: "unknown"}
	assert.Equal(t, "unexpected chunk type: unknown", unknownChunk.String())
}

func TestCallWithText(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	var receivedText string

	callback := func(_ context.Context, chunk Chunk) error {
		assert.Equal(t, ChunkTypeText, chunk.Type)
		receivedText = chunk.Content
		return nil
	}

	// Test with valid text
	err := CallWithText(ctx, callback, "Hello, world!")
	require.NoError(t, err)
	assert.Equal(t, "Hello, world!", receivedText)

	// Test with empty text
	err = CallWithText(ctx, callback, "")
	require.NoError(t, err)
	assert.Equal(t, "Hello, world!", receivedText) // Should not change

	// Test with nil callback
	err = CallWithText(ctx, nil, "Hello, world!")
	require.NoError(t, err)

	// Test with error from callback
	expectedErr := errors.New("callback error")
	errorCallback := func(_ context.Context, _ Chunk) error {
		return expectedErr
	}
	err = CallWithText(ctx, errorCallback, "Hello, world!")
	require.Error(t, err)
	assert.Equal(t, expectedErr, err)
}

//nolint:funlen
func TestCallWithReasoning(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	var receivedReasoning string

	callback := func(_ context.Context, chunk Chunk) error {
		assert.Equal(t, ChunkTypeReasoning, chunk.Type)
		receivedReasoning = chunk.ReasoningContent
		return nil
	}

	// Test with valid reasoning
	err := CallWithReasoning(ctx, callback, "Step 1: Analyze.")
	require.NoError(t, err)
	assert.Equal(t, "Step 1: Analyze.", receivedReasoning)

	// Test with empty reasoning
	err = CallWithReasoning(ctx, callback, "")
	require.NoError(t, err)
	assert.Equal(t, "Step 1: Analyze.", receivedReasoning) // Should not change

	// Test with nil callback
	err = CallWithReasoning(ctx, nil, "Step 1: Analyze.")
	require.NoError(t, err)

	// Test with multiline reasoning
	multilineReasoning := "Step 1: Define the problem.\nStep 2: Collect data.\nStep 3: Form a hypothesis."
	err = CallWithReasoning(ctx, callback, multilineReasoning)
	require.NoError(t, err)
	assert.Equal(t, multilineReasoning, receivedReasoning)

	// Test with special characters and Unicode
	specialCharsReasoning := "分析步骤: 检查数据 ¥€$£ß"
	err = CallWithReasoning(ctx, callback, specialCharsReasoning)
	require.NoError(t, err)
	assert.Equal(t, specialCharsReasoning, receivedReasoning)

	// Test with very long reasoning text
	longReasoning := "Step 1: " + strings.Repeat("Analysis and evaluation. ", 100)
	err = CallWithReasoning(ctx, callback, longReasoning)
	require.NoError(t, err)
	assert.Equal(t, longReasoning, receivedReasoning)

	// Test multiple sequential calls with accumulation
	accumulatedReasoning := ""
	accumulationCallback := func(_ context.Context, chunk Chunk) error {
		assert.Equal(t, ChunkTypeReasoning, chunk.Type)
		accumulatedReasoning += chunk.ReasoningContent
		return nil
	}

	err = CallWithReasoning(ctx, accumulationCallback, "First part. ")
	require.NoError(t, err)
	err = CallWithReasoning(ctx, accumulationCallback, "Second part. ")
	require.NoError(t, err)
	err = CallWithReasoning(ctx, accumulationCallback, "Final part.")
	require.NoError(t, err)

	assert.Equal(t, "First part. Second part. Final part.", accumulatedReasoning)

	// Test with error from callback
	expectedErr := errors.New("callback error")
	errorCallback := func(_ context.Context, _ Chunk) error {
		return expectedErr
	}
	err = CallWithReasoning(ctx, errorCallback, "Step 1: Analyze.")
	require.Error(t, err)
	assert.Equal(t, expectedErr, err)
}

func TestCallWithToolCall(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	var receivedToolCall ToolCall

	callback := func(_ context.Context, chunk Chunk) error {
		assert.Equal(t, ChunkTypeToolCall, chunk.Type)
		receivedToolCall = chunk.ToolCall
		return nil
	}

	toolCall := NewToolCall("123", "weather", `{"location": "New York"}`)

	// Test with valid tool call
	err := CallWithToolCall(ctx, callback, toolCall)
	require.NoError(t, err)
	assert.Equal(t, toolCall, receivedToolCall)

	// Test with missing ID
	invalidToolCall := NewToolCall("", "weather", `{"location": "New York"}`)
	err = CallWithToolCall(ctx, callback, invalidToolCall)
	require.Error(t, err)
	assert.Equal(t, ErrToolCallIDRequired, err)

	// Test with missing Name
	invalidToolCall = NewToolCall("123", "", `{"location": "New York"}`)
	err = CallWithToolCall(ctx, callback, invalidToolCall)
	require.Error(t, err)
	assert.Equal(t, ErrToolCallNameRequired, err)

	// Test with nil callback
	err = CallWithToolCall(ctx, nil, toolCall)
	require.NoError(t, err)

	// Test with error from callback
	expectedErr := errors.New("callback error")
	errorCallback := func(_ context.Context, _ Chunk) error {
		return expectedErr
	}
	err = CallWithToolCall(ctx, errorCallback, toolCall)
	require.Error(t, err)
	assert.Equal(t, expectedErr, err)
}

func TestAppendToolCall(t *testing.T) {
	t.Parallel()

	src := NewToolCall("123", "weather", `{"location": "New York"}`)
	dst := ToolCall{
		ID:        "",
		Name:      "",
		Arguments: "",
	}

	AppendToolCall(src, &dst)

	assert.Equal(t, "123", dst.ID)
	assert.Equal(t, "weather", dst.Name)
	assert.Equal(t, `{"location": "New York"}`, dst.Arguments)

	// Test appending to existing arguments
	src = NewToolCall("123", "weather", `, "unit": "celsius"}`)
	AppendToolCall(src, &dst)

	assert.Equal(t, "123", dst.ID)
	assert.Equal(t, "weather", dst.Name)
	assert.Equal(t, `{"location": "New York"}, "unit": "celsius"}`, dst.Arguments)
}

func TestCallWithDone(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	var receivedDone bool

	callback := func(_ context.Context, chunk Chunk) error {
		assert.Equal(t, ChunkTypeDone, chunk.Type)
		receivedDone = true
		return nil
	}

	// Test with valid callback
	err := CallWithDone(ctx, callback)
	require.NoError(t, err)
	assert.True(t, receivedDone)

	// Test with nil callback
	err = CallWithDone(ctx, nil)
	require.NoError(t, err)

	// Test with error from callback
	expectedErr := errors.New("callback error")
	errorCallback := func(_ context.Context, _ Chunk) error {
		return expectedErr
	}
	err = CallWithDone(ctx, errorCallback)
	require.Error(t, err)
	assert.Equal(t, expectedErr, err)
}

//nolint:funlen
func TestIntegration(t *testing.T) {
	t.Parallel()

	ctx := t.Context()

	// Track received chunks
	var textChunks []string
	var reasoningChunks []string
	var toolCalls []ToolCall
	var doneReceived bool

	callback := func(_ context.Context, chunk Chunk) error {
		switch chunk.Type {
		case ChunkTypeNone:
			// Just ensure we can handle this type
		case ChunkTypeText:
			textChunks = append(textChunks, chunk.Content)
		case ChunkTypeReasoning:
			reasoningChunks = append(reasoningChunks, chunk.ReasoningContent)
		case ChunkTypeToolCall:
			toolCalls = append(toolCalls, chunk.ToolCall)
		case ChunkTypeDone:
			doneReceived = true
		}
		return nil
	}

	// Simulate streaming text chunks
	_ = CallWithText(ctx, callback, "Hello")
	_ = CallWithText(ctx, callback, ", ")
	_ = CallWithText(ctx, callback, "world!")

	// Simulate streaming reasoning chunks
	_ = CallWithReasoning(ctx, callback, "Step 1: Start with a greeting.")
	_ = CallWithReasoning(ctx, callback, "Step 2: Add punctuation.")
	_ = CallWithReasoning(ctx, callback, "Step 3: Complete the phrase.")

	// Simulate streaming tool calls
	weatherTool := NewToolCall("123", "weather", `{"location": `)
	_ = CallWithToolCall(ctx, callback, weatherTool)

	weatherTool2 := NewToolCall("123", "weather", `"New York"}`)
	_ = CallWithToolCall(ctx, callback, weatherTool2)

	timeTool := NewToolCall("456", "getTime", `{}`)
	_ = CallWithToolCall(ctx, callback, timeTool)

	// Signal stream completion
	_ = CallWithDone(ctx, callback)

	// Verify results
	assert.Equal(t, []string{"Hello", ", ", "world!"}, textChunks)
	assert.Equal(t, []string{
		"Step 1: Start with a greeting.",
		"Step 2: Add punctuation.",
		"Step 3: Complete the phrase.",
	}, reasoningChunks)

	require.Len(t, toolCalls, 3)
	assert.Equal(t, "123", toolCalls[0].ID)
	assert.Equal(t, "weather", toolCalls[0].Name)
	assert.Equal(t, `{"location": `, toolCalls[0].Arguments)

	assert.Equal(t, "123", toolCalls[1].ID)
	assert.Equal(t, "weather", toolCalls[1].Name)
	assert.Equal(t, `"New York"}`, toolCalls[1].Arguments)

	assert.Equal(t, "456", toolCalls[2].ID)
	assert.Equal(t, "getTime", toolCalls[2].Name)
	assert.Equal(t, `{}`, toolCalls[2].Arguments)

	// Verify done was received
	assert.True(t, doneReceived)
}

func TestToolCallMarshalUnmarshal(t *testing.T) {
	t.Parallel()

	toolCall := NewToolCall("123", "weather", `{"location": "New York", "units": "celsius"}`)

	// Test marshaling
	data, err := json.Marshal(toolCall)
	require.NoError(t, err)

	// Test unmarshaling
	var unmarshaled ToolCall
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)

	assert.Equal(t, toolCall, unmarshaled)
}

func TestChunkMarshalUnmarshal(t *testing.T) {
	t.Parallel()

	// Test text chunk
	textChunk := NewTextChunk("Hello, world!")
	data, err := json.Marshal(textChunk)
	require.NoError(t, err)

	var unmarshaled Chunk
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)
	assert.Equal(t, textChunk, unmarshaled)

	// Test reasoning chunk
	reasoningChunk := NewReasoningChunk("Step 1: Analyze.")
	data, err = json.Marshal(reasoningChunk)
	require.NoError(t, err)

	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)
	assert.Equal(t, reasoningChunk, unmarshaled)

	// Test tool call chunk
	toolCall := NewToolCall("123", "weather", `{"location": "New York"}`)
	toolCallChunk := NewToolCallChunk(toolCall)
	data, err = json.Marshal(toolCallChunk)
	require.NoError(t, err)

	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)
	assert.Equal(t, toolCallChunk, unmarshaled)

	// Test done chunk
	doneChunk := NewDoneChunk()
	data, err = json.Marshal(doneChunk)
	require.NoError(t, err)

	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)
	assert.Equal(t, doneChunk, unmarshaled)
}
