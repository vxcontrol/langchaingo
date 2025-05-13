package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func main() {
	key := os.Getenv("NVIDIA_API_KEY")
	llm, err := openai.New(
		openai.WithBaseURL("https://integrate.api.nvidia.com/v1/"),
		openai.WithModel("mistralai/mixtral-8x7b-instruct-v0.1"),
		openai.WithToken(key),
		// openai.WithHTTPClient(httputil.DebugHTTPClient),
	)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	content := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "You are a golang expert"),
		llms.TextParts(llms.ChatMessageTypeHuman, "explain why go is a great fit for ai based products"),
	}

	if _, err = llm.GenerateContent(ctx, content,
		llms.WithMaxTokens(4096),
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			fmt.Println(chunk.String())
			return nil
		})); err != nil {
		log.Fatal(err)
	}
}
