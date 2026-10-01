# Extended Capabilities Example

This example demonstrates Claude's extended thinking on a task that also needs a long answer:
- **Extended Thinking**: Deep reasoning before the answer
- **Long Output**: A comprehensive answer in a single response

## Features Demonstrated

1. **Thinking and a long answer together**: Shows how to combine extended thinking with a high output token limit
2. **Complex Task**: Generates a comprehensive distributed systems guide requiring both deep reasoning and extensive output
3. **Token Metrics**: Displays detailed token usage including thinking tokens and output tokens

## Running the Example

```bash
# Set your API key
export ANTHROPIC_API_KEY=your-api-key

# Run the example
go run .
```

## Key Implementation Points

```go
opts := []llms.CallOption{
    // Extended thinking for complex reasoning
    llms.WithReasoning(llms.ReasoningHigh, 16000),

    // Leave room for a long answer after the thinking
    llms.WithMaxTokens(32000),
}
```

## What to Expect

- The model will use extended thinking to reason about the complex distributed systems topic
- It will generate a comprehensive guide in one response
- Token metrics will show both thinking tokens used and total output generated
- If the response is large (>10K chars), you'll have the option to save it to a file

## Requirements

- Claude Sonnet 5.5 model (`claude-sonnet-5-5`)
- Valid Anthropic API key
