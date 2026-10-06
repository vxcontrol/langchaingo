# Bedrock Provider Example

This example demonstrates how to use the Bedrock LLM with different model providers, including support for Nova models and inference profiles.

## Features

- Automatic provider detection from model ID
- Support for Nova models (e.g., `amazon.nova-lite-v1:0`)
- Support for inference profiles (e.g., `us.amazon.nova-lite-v1:0`)

## Prerequisites

1. AWS credentials configured (via environment variables or AWS credentials file)
2. Access to the Bedrock models you want to use

## Usage

```bash
# Using the default Nova Lite model
go run main.go

# Using inference profile
go run main.go -model "us.amazon.nova-lite-v1:0"

# Using an Anthropic model
go run main.go -model "us.anthropic.claude-sonnet-5-5"

# Custom prompt
go run main.go -prompt "What is the capital of France?"

# Verbose output
go run main.go -verbose
```

## Environment Variables

Set these environment variables before running:

```bash
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

## Supported Providers

The Bedrock integration automatically detects the provider from the model ID:

- **Nova**: Models containing `.nova-` (e.g., `amazon.nova-lite-v1:0`, `us.amazon.nova-pro-v1:0`)
- **Anthropic**: Models containing `anthropic` (e.g., `anthropic.claude-sonnet-5-5`)
- **Amazon**: Models containing `amazon` (excluding Nova)
- **Meta**: Models containing `meta` (e.g., `meta.llama3-3-70b-instruct-v1:0`)
- **AI21**: Models containing `ai21` (e.g., `ai21.jamba-1-5-large-v1:0`)
