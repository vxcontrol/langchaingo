This directory contains langchaingo provider for Google's models.

* In the main `googleai` directory: provider for Google AI
  (https://ai.google.dev/)
* In the `vertex` directory: provider for GCP Vertex AI
  (https://cloud.google.com/vertex-ai/)
* In the `palm` directory: provider for the legacy PaLM models.

Both the `googleai` and `vertex` providers give access to Gemini-family
multi-modal LLMs, and both now run on one SDK, `google.golang.org/genai`, which
reaches either backend from the same client.

The `vertex` package is therefore a thin wrapper: it fixes the backend to Vertex
AI, takes the cloud project and location from options or from
`GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`, and refuses to start without
them. Every capability — generation, streaming, tools, thinking, structured
output, embeddings — comes from the `googleai` door, so the two cannot drift
apart. The package used to carry its own copy of that adapter on the deprecated
`cloud.google.com/go/vertexai/genai` SDK, along with a code generator; both are
gone.

----

Testing:

The test code between `googleai` and `vertex` is also shared, and lives in
the `shared_test` directory. The same tests are run for both providers.
