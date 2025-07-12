# Chroma Support

You can access Chroma via the included implementation of the
[`vectorstores.VectorStore` interface](../vectorstores.go)
by creating and using a Chroma client `Store` instance with
the [`New` function](./chroma.go) API.

## Client/Server

Until
[an "in-memory" version](https://docs.trychroma.com/usage-guide#running-chroma-in-clientserver-mode)
is released, only client/server mode is available.

> **Note:** Additional ways to run Chroma locally can be found
> in [Chroma Cookbook](https://cookbook.chromadb.dev/running/running-chroma/)

Use the [`WithChromaURL` API](./options.go) or the `CHROMA_URL` environment
variable to specify the URL of the Chroma server when creating the client instance.

## Configuration Options

The Chroma vector store supports several configuration options:

- **`WithChromaURL(url string)`** - Specifies the Chroma server URL
- **`WithEmbedder(embedder embeddings.Embedder)`** - Sets the embedding function to use
- **`WithNameSpace(namespace string)`** - Sets the namespace for document isolation
- **`WithDistanceFunction(func chromatypes.DistanceFunction)`** - Sets the distance function (default: L2)
- **`WithIncludes(includes []chromatypes.QueryEnum)`** - Specifies which data to include in query results

### Distance Functions

The following distance functions are supported:
- `chromatypes.L2` (default) - Euclidean distance
- `chromatypes.COSINE` - Cosine similarity
- `chromatypes.IP` - Inner product

## Using with OpenAI Embeddings

To use OpenAI embeddings with Chroma, create an embedder and pass it to the store:

```go
import (
    "github.com/vxcontrol/langchaingo/embeddings"
    "github.com/vxcontrol/langchaingo/llms/openai"
    "github.com/vxcontrol/langchaingo/vectorstores/chroma"
)

// Create OpenAI LLM
llm, err := openai.New(
    openai.WithToken(os.Getenv("OPENAI_API_KEY")),
    openai.WithEmbeddingModel("text-embedding-ada-002"),
)

// Create embedder
embedder, err := embeddings.NewEmbedder(llm)

// Create Chroma store
store, err := chroma.New(
    chroma.WithChromaURL(os.Getenv("CHROMA_URL")),
    chroma.WithEmbedder(embedder),
    chroma.WithDistanceFunction(chromatypes.COSINE),
)
```

## Features

### Document Management
- Add documents with metadata
- Automatic document ID generation
- Support for custom metadata fields

### Similarity Search
- Similarity search with configurable result count
- Score threshold filtering
- Metadata filtering with complex queries
- Support for various filter operators (`$eq`, `$in`, `$and`, `$gte`, etc.)

### Metadata Filtering

The store supports sophisticated metadata filtering:

```go
// Simple equality filter
filter := map[string]any{
    "location": map[string]any{"$eq": "tokyo"},
}

// Complex AND filter with multiple conditions
filter := map[string]any{
    "$and": []map[string]any{
        {"area": map[string]any{"$gte": 1000}},
        {"population": map[string]any{"$gte": 13}},
    },
}

docs, err := store.SimilaritySearch(ctx, query, numDocs, 
    vectorstores.WithFilters(filter),
    vectorstores.WithScoreThreshold(0.8),
)
```

## Running With Docker

Running a Chroma server in a local docker instance can be especially useful for testing
and development workflows. An example invocation scenario is presented below:

### Starting the Chroma Server

As of this writing, the recommended Chroma docker image is:

```shell
$ docker run -p 8000:8000 ghcr.io/chroma-core/chroma:0.5.0
```

### Running an Example `langchaingo` Application

With the Chroma server running, you can run the included example:

```shell
$ export CHROMA_URL=http://localhost:8000
$ export OPENAI_API_KEY=YourOpenApiKeyGoesHere
$ go run ./examples/chroma-vectorstore-example/chroma_vectorstore_example.go
Results:
1. case: Up to 5 Cities in Japan
    result: Tokyo, Nagoya, Kyoto, Fukuoka, Hiroshima
2. case: A City in South America
    result: Buenos Aires
3. case: Large Cities in South America
    result: Sao Paulo, Rio de Janeiro
```

## Example Usage

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/vxcontrol/langchaingo/embeddings"
    "github.com/vxcontrol/langchaingo/llms/openai"
    "github.com/vxcontrol/langchaingo/schema"
    "github.com/vxcontrol/langchaingo/vectorstores"
    "github.com/vxcontrol/langchaingo/vectorstores/chroma"
    
    chromatypes "github.com/amikos-tech/chroma-go/types"
    "github.com/google/uuid"
)

func main() {
    // Create OpenAI LLM and embedder
    llm, err := openai.New(
        openai.WithToken(os.Getenv("OPENAI_API_KEY")),
        openai.WithEmbeddingModel("text-embedding-ada-002"),
    )
    if err != nil {
        log.Fatal(err)
    }
    
    embedder, err := embeddings.NewEmbedder(llm)
    if err != nil {
        log.Fatal(err)
    }

    // Create Chroma store
    store, err := chroma.New(
        chroma.WithChromaURL(os.Getenv("CHROMA_URL")),
        chroma.WithEmbedder(embedder),
        chroma.WithDistanceFunction(chromatypes.COSINE),
        chroma.WithNameSpace(uuid.New().String()),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Add documents
    docs := []schema.Document{
        {
            PageContent: "Tokyo",
            Metadata: map[string]any{
                "country": "Japan",
                "population": 9.7,
                "area": 622,
            },
        },
        {
            PageContent: "Paris",
            Metadata: map[string]any{
                "country": "France",
                "population": 11,
                "area": 105,
            },
        },
    }

    _, err = store.AddDocuments(context.Background(), docs)
    if err != nil {
        log.Fatal(err)
    }

    // Search for similar documents
    results, err := store.SimilaritySearch(
        context.Background(),
        "Which cities are in Japan?",
        5,
        vectorstores.WithScoreThreshold(0.8),
    )
    if err != nil {
        log.Fatal(err)
    }

    for _, doc := range results {
        fmt.Printf("Content: %s, Score: %.2f\n", doc.PageContent, doc.Score)
    }
}
```

## Tests

The test suite `chroma_test.go` provides comprehensive coverage including:
- Basic document operations
- Similarity search with score thresholds
- Metadata filtering with various operators
- Integration with retrieval chains
- Error handling and edge cases

The tests use both environment variables for configuration and Docker containers
for automated testing environments.

## Dependencies

This implementation uses:
- [`github.com/amikos-tech/chroma-go`](https://github.com/amikos-tech/chroma-go) - Official Chroma Go client
- [`github.com/google/uuid`](https://github.com/google/uuid) - UUID generation for document IDs
