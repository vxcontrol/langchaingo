package main

import (
	"context"
	"fmt"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/llamafile"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func main() {
	options := []llamafile.Option{
		llamafile.WithEmbeddingSize(2048),
		llamafile.WithTemperature(0.8),
	}
	llm, err := llamafile.New(options...)

	if err != nil {
		panic(err)
	}

	parts := []llms.ContentPart{
		llms.TextContent{Text: "Brazil is a country? answer yes or no"},
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	_, err = llm.GenerateContent(
		context.Background(),
		content,
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			fmt.Println(chunk.String())
			return nil
		}),
	)

	if err != nil {
		panic(err)
	}
}
