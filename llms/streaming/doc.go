// Package streaming provides a streaming interface for LLMs.
//
// This package implements utilities for handling streaming responses from language models,
// allowing for real-time processing of generated content. It supports five types of chunks:
// text chunks for regular content, reasoning chunks for step-by-step thinking processes,
// tool call chunks for function/tool invocations, none chunks as a placeholder,
// and done chunks to signal the end of a streaming session.
//
// The streaming interface uses a callback-based approach where each chunk is processed
// as it arrives, enabling applications to display partial results and interactive responses.
// This is particularly useful for long-form content generation, reasoning steps visualization,
// and tool-using agents.
//
// Basic usage involves registering a callback function that handles chunks as they arrive:
//
//	streamingFunc := func(ctx context.Context, chunk streaming.Chunk) error {
//		switch chunk.Type {
//		case streaming.ChunkTypeNone:
//			// It's not expected to receive this chunk type
//			// but it's not an error to receive it
//			return nil
//		case streaming.ChunkTypeText:
//			// Process text content
//			fmt.Print(chunk.Content)
//		case streaming.ChunkTypeReasoning:
//			// Process reasoning/thinking content
//			fmt.Print(chunk.ReasoningContent)
//		case streaming.ChunkTypeToolCall:
//			// Process tool call
//			fmt.Printf("Tool call: %s\n", chunk.ToolCall.String())
//		case streaming.ChunkTypeDone:
//			// Streaming is done - this signals the end of the stream
//			// and can be used to finalize processing
//			fmt.Println("Streaming is done")
//		}
//		return nil
//	}
//
// The callback function can be passed to LLM implementations that support streaming.
// When the LLM finishes generating all content, it should call the streaming.CallWithDone
// function to signal the end of the streaming session, allowing consumers to perform
// any necessary cleanup or finalization.
package streaming
