# Chroma Vector Store Example

This example demonstrates how to use the Chroma vector store with LangChain in Go. It showcases various operations and queries on a vector store containing information about cities from around the world.

## Prerequisites

Before running this example, you need to set the following environment variables:

- `CHROMA_URL`: The URL of your Chroma server (e.g., `http://localhost:8000`)
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings

## What This Example Does

1. **Vector Store Creation**: The example creates a new Chroma vector store with the following configuration:
   - Uses OpenAI's `text-embedding-ada-002` model for embeddings
   - Applies COSINE distance function for similarity calculations
   - Uses a random UUID-based namespace for document isolation
   - Connects to Chroma server via the `CHROMA_URL` environment variable

2. **Adding Documents**: It adds 13 documents to the vector store, representing cities from different countries:
   - **Japanese cities**: Tokyo, Kyoto, Hiroshima, Kazuno, Nagoya, Toyota, Fukuoka
   - **European cities**: Paris, London
   - **South American cities**: Santiago, Buenos Aires, Rio de Janeiro, Sao Paulo
   
   Each document includes metadata with population (in millions) and area (in square kilometers).

3. **Similarity Searches**: The example performs three different similarity searches with distinct strategies:

   a. **Up to 5 Cities in Japan**: 
      - Query: "Which of these are cities are located in Japan?"
      - Limits results to 5 documents
      - Uses score threshold of 0.8 to filter low-relevance matches
   
   b. **A City in South America**: 
      - Query: "Which of these are cities are located in South America?"
      - Limits results to 1 document
      - Uses score threshold of 0.8 to ensure high relevance
   
   c. **Large Cities in South America**: 
      - Query: "Which of these are cities are located in South America?"
      - Uses metadata filters instead of score threshold:
        - Area >= 1000 square kilometers
        - Population >= 13 million inhabitants
      - Demonstrates complex filtering with `$and` operator

4. **Result Display**: Finally, it prints out the results of each search, showing the matching cities for each query.

## Key Features

- **OpenAI Integration**: Uses OpenAI embeddings for semantic understanding
- **Flexible Configuration**: Environment-based configuration for easy deployment
- **Document Metadata**: Demonstrates adding and using structured metadata
- **Multiple Search Strategies**: Shows different approaches to similarity search:
  - Score threshold filtering for relevance control
  - Metadata filtering for attribute-based queries
  - Result count limiting for focused results
- **Distance Functions**: Uses COSINE similarity for better semantic matching
- **Namespace Isolation**: Uses UUID-based namespaces for document organization

## Running the Example

1. Start a Chroma server:
   ```bash
   docker run -p 8000:8000 ghcr.io/chroma-core/chroma:0.5.0
   ```

2. Set environment variables:
   ```bash
   export CHROMA_URL=http://localhost:8000
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Run the example:
   ```bash
   go run chroma_vectorstore_example.go
   ```

Expected output:
```
Results:
1. case: Up to 5 Cities in Japan
    result: Tokyo, Nagoya, Kyoto, Fukuoka, Hiroshima
2. case: A City in South America
    result: Buenos Aires
3. case: Large Cities in South America
    result: Sao Paulo, Rio de Janeiro
```

This example is excellent for developers looking to understand how to integrate and use vector stores in their Go applications, particularly for semantic search and similarity matching tasks with complex filtering requirements.
