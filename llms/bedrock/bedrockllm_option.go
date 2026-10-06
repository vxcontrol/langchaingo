package bedrock

import (
	"time"

	"github.com/vxcontrol/langchaingo/callbacks"
	"github.com/vxcontrol/langchaingo/llms"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// Option is an option for the Bedrock LLM.
type Option func(*options)

type options struct {
	modelProvider     string
	modelID           string
	client            *bedrockruntime.Client
	callbackHandler   callbacks.Handler
	useConverseAPI    bool
	enableAutoCaching bool
}

// WithModel allows setting a custom modelId.
//
// If not set, ModelAnthropicClaudeHaiku45 is used.
func WithModel(modelID string) Option {
	return func(o *options) {
		o.modelID = modelID
	}
}

// WithModelProvider allows setting a custom model provider.
//
// Deprecated: the door detects the provider from the model ID, so this option
// has no effect. For an ID the InvokeModel path cannot place, use WithConverseAPI.
func WithModelProvider(modelProvider string) Option {
	return func(o *options) {
		o.modelProvider = modelProvider
	}
}

// WithClient allows setting a custom bedrockruntime.Client.
//
// You may use this to pass a custom bedrockruntime.Client
// with custom configuration options
// such as setting custom credentials, region, endpoint, etc.
//
// By default, a new client will be created using the default credentials chain.
func WithClient(client *bedrockruntime.Client) Option {
	return func(o *options) {
		o.client = client
	}
}

// WithCallback allows setting a custom Callback Handler.
func WithCallback(callbackHandler callbacks.Handler) Option {
	return func(o *options) {
		o.callbackHandler = callbackHandler
	}
}

// WithConverseAPI enables the use of the unified Bedrock Converse API
// instead of the model-specific legacy implementations.
//
// Through Converse the door sends tool calls, streams with ConverseStream,
// sends reasoning settings to the Claude, Nova 2 Lite and GPT OSS models it
// knows, takes text and image input, and places cache points. It serves every
// model in models_list.go. Cache token counts arrive in each choice's
// GenerationInfo as CacheReadInputTokens and CacheCreationInputTokens.
//
// Note: This is the recommended approach for new applications.
func WithConverseAPI() Option {
	return func(o *options) {
		o.useConverseAPI = true
	}
}

// WithAutomaticCaching enables automatic prompt caching for Claude models whose
// IDs contain claude-opus-4, claude-sonnet-4, claude-haiku-4 or a 5.x Opus,
// Sonnet, Haiku, Fable or Mythos name.
//
// The door marks the last assistant or tool-result message before the new user
// turn with a 5-minute cache point. Converse carries that mark on an assistant
// message only and adds cache points after the system prompt and at the end of
// the final message; the InvokeModel path sends the system prompt uncached.
func WithAutomaticCaching() Option {
	return func(o *options) {
		o.enableAutoCaching = true
	}
}

// EphemeralCache creates a standard ephemeral cache control for Bedrock with 5-minute duration.
func EphemeralCache() *llms.CacheControl {
	return &llms.CacheControl{
		Type:     "ephemeral",
		Duration: 5 * time.Minute,
	}
}

// EphemeralCacheOneHour creates a 1-hour ephemeral cache control for Bedrock.
func EphemeralCacheOneHour() *llms.CacheControl {
	return &llms.CacheControl{
		Type:     "ephemeral",
		Duration: time.Hour,
	}
}

// CachedContent represents content with caching instructions for Bedrock.
// This wraps any ContentPart and adds cache control metadata.
//
// Note: For most use cases, prefer the bedrock.WithAutomaticCaching() option.
// This manual wrapper is only needed for fine-grained cache control.
type CachedContent struct {
	llms.ContentPart
	CacheControl *llms.CacheControl `json:"cache_control,omitempty"`
}

// WithCacheControl wraps content with cache control instructions for Bedrock.
// This allows explicit control over what content should be cached.
//
// Recommended: Use bedrock.WithAutomaticCaching() option instead for transparent caching.
//
// Manual usage (when fine-grained control is needed):
//
//	bedrock.WithCacheControl(
//	    llms.TextPart("long context..."),
//	    bedrock.EphemeralCache(),
//	)
//
// The door sends the cache point for any model; the model card says whether
// the model caches.
func WithCacheControl(content llms.ContentPart, control *llms.CacheControl) CachedContent {
	return CachedContent{
		ContentPart:  content,
		CacheControl: control,
	}
}
