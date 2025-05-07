module github.com/vxcontrol/langchaingo/examples/ollama-chroma-vectorstore-example

go 1.24.0

require (
	github.com/google/uuid v1.6.0
	github.com/vxcontrol/langchaingo v0.1.13-update.0
)

require (
	github.com/Masterminds/semver v1.5.0 // indirect
	github.com/amikos-tech/chroma-go v0.1.2 // indirect
	github.com/dlclark/regexp2 v1.11.4 // indirect
	github.com/oklog/ulid v1.3.1 // indirect
	github.com/ollama/ollama v0.6.8 // indirect
	github.com/pkoukk/tiktoken-go v0.1.6 // indirect
)

replace github.com/vxcontrol/langchaingo => ../..
