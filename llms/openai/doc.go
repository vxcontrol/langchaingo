// Package openai provides an interface to OpenAI's language models.
//
// # Token Limits
//
// For setting token limits with OpenAI models, use openai.WithMaxCompletionTokens()
// for clarity. The OpenAI API now uses max_completion_tokens as the field for
// limiting output tokens.
//
//	// Recommended for clarity:
//	llm.GenerateContent(ctx, messages,
//	    openai.WithMaxCompletionTokens(100),
//	)
//
//	// Also works (backward compatible):
//	llm.GenerateContent(ctx, messages,
//	    llms.WithMaxTokens(100),
//	)
//
// Both options set the same underlying field. By default, the implementation sends
// max_completion_tokens (modern field). For older OpenAI-compatible servers that
// only support max_tokens, use WithLegacyMaxTokensField():
//
//	llm.GenerateContent(ctx, messages,
//	    llms.WithMaxTokens(100),
//	    openai.WithLegacyMaxTokensField(), // Forces use of max_tokens field
//	)
//
// # API Keys
//
// An API key is required when the base URL is empty, meaning the default
// OpenAI endpoint; when its host is one of the known public OpenAI-compatible
// providers (see APIKeyRequiredBaseURLs); when the host sits in a
// tenant-specific Azure OpenAI or OpenAI data-residency zone; or whenever
// APIType is Azure or Azure AD, whatever the base URL. Local servers such as
// vLLM, Ollama, llama.cpp, and SGLang can be used without a key.
package openai
