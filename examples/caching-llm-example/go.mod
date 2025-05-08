module github.com/vxcontrol/langchaingo/examples/caching-llm-example

go 1.24.0

require (
	github.com/mitchellh/go-wordwrap v1.0.1
	github.com/vxcontrol/langchaingo v0.1.13-update.1
)

require (
	github.com/Code-Hex/go-generics-cache v1.3.1 // indirect
	github.com/dlclark/regexp2 v1.11.4 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/ollama/ollama v0.6.8 // indirect
	github.com/pkoukk/tiktoken-go v0.1.6 // indirect
	golang.org/x/exp v0.0.0-20250218142911-aa4b98e5adaa // indirect
)

replace github.com/vxcontrol/langchaingo => ../..
