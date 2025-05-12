package main

import (
	"context"
	"fmt"
	"log"

	"github.com/vxcontrol/langchaingo/embeddings"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/ernie"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func main() {
	llm, err := ernie.New(ernie.WithModelName(ernie.ModelNameERNIEBot))
	// note:
	// You would include ernie.WithAKSK(apiKey,secretKey) to use specific auth info.
	// You would include ernie.WithModelName(ernie.ModelNameERNIEBot) to use the ERNIE-Bot model.
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	completion, err := llms.GenerateFromSinglePrompt(ctx, llm, "介绍一下你自己",
		llms.WithTemperature(0.8),
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			fmt.Println(chunk.String())
			return nil
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	_ = completion

	// embedding
	embedding, _ := embeddings.NewEmbedder(llm)

	emb, err := embedding.EmbedDocuments(ctx, []string{"你好"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Embedding-V1:", len(emb), len(emb[0]))
}
