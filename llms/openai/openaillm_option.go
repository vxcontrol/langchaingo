package openai

import (
	"github.com/vxcontrol/langchaingo/callbacks"
	"github.com/vxcontrol/langchaingo/llms/openai/internal/openaiclient"
)

const (
	tokenEnvVarName        = "OPENAI_API_KEY"      //nolint:gosec
	modelEnvVarName        = "OPENAI_MODEL"        //nolint:gosec
	baseURLEnvVarName      = "OPENAI_BASE_URL"     //nolint:gosec
	baseAPIBaseEnvVarName  = "OPENAI_API_BASE"     //nolint:gosec
	organizationEnvVarName = "OPENAI_ORGANIZATION" //nolint:gosec
)

type APIType openaiclient.APIType

const (
	APITypeOpenAI  APIType = APIType(openaiclient.APITypeOpenAI)
	APITypeAzure           = APIType(openaiclient.APITypeAzure)
	APITypeAzureAD         = APIType(openaiclient.APITypeAzureAD)
)

const (
	DefaultAPIVersion = "2023-05-15"
)

type options struct {
	token        string
	model        string
	baseURL      string
	organization string
	apiType      APIType
	httpClient   openaiclient.Doer

	responseFormat *ResponseFormat

	// fine tuning reasoning options for various LLM providers
	useReasoningMaxTokens bool
	modernReasoningFormat bool

	// preserve reasoning content in multi-turn conversations with tool calls
	preserveReasoningContent bool

	// carry a structured-output schema in the prompt for vendors without json_schema
	structuredOutputFallback bool

	// required when APIType is APITypeAzure or APITypeAzureAD
	apiVersion          string
	embeddingModel      string
	embeddingDimensions int

	callbackHandler callbacks.Handler
}

// Option is a functional option for the OpenAI client.
type Option func(*options)

// ResponseFormat is the response format for the OpenAI client.
type ResponseFormat = openaiclient.ResponseFormat

// ResponseFormatJSONSchema is the JSON Schema response format in structured output.
type ResponseFormatJSONSchema = openaiclient.ResponseFormatJSONSchema

// ResponseFormatJSONSchemaProperty is the JSON Schema property in structured output.
type ResponseFormatJSONSchemaProperty = openaiclient.ResponseFormatJSONSchemaProperty

// ResponseFormatJSON is the JSON response format.
var ResponseFormatJSON = &ResponseFormat{Type: "json_object"} //nolint:gochecknoglobals

// WithToken passes the OpenAI API token to the client. If not set, the token
// is read from the OPENAI_API_KEY environment variable. A token is required
// only for known public OpenAI-compatible providers; local backends such as
// vLLM may be used without one.
func WithToken(token string) Option {
	return func(opts *options) {
		opts.token = token
	}
}

// WithModel passes the OpenAI model to the client. If not set, the model
// is read from the OPENAI_MODEL environment variable.
// Required when ApiType is Azure.
func WithModel(model string) Option {
	return func(opts *options) {
		opts.model = model
	}
}

// WithEmbeddingModel passes the OpenAI model to the client. Required when ApiType is Azure.
func WithEmbeddingModel(embeddingModel string) Option {
	return func(opts *options) {
		opts.embeddingModel = embeddingModel
	}
}

// WithEmbeddingDimensions passes the OpenAI embeddings dimensions to the client.
// Requires a compatible model, test-embedding-3 or later.
// For more info, please check openai doc
// https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions
func WithEmbeddingDimensions(dimensions int) Option {
	return func(opts *options) {
		opts.embeddingDimensions = dimensions
	}
}

// WithBaseURL passes the OpenAI base url to the client. If not set, the base url
// is read from the OPENAI_BASE_URL environment variable. If still not set in ENV
// VAR OPENAI_BASE_URL, then the default value is https://api.openai.com/v1 is used.
func WithBaseURL(baseURL string) Option {
	return func(opts *options) {
		opts.baseURL = baseURL
	}
}

// WithOrganization passes the OpenAI organization to the client. If not set, the
// organization is read from the OPENAI_ORGANIZATION.
func WithOrganization(organization string) Option {
	return func(opts *options) {
		opts.organization = organization
	}
}

// WithAPIType passes the api type to the client. If not set, the default value
// is APITypeOpenAI.
func WithAPIType(apiType APIType) Option {
	return func(opts *options) {
		opts.apiType = apiType
	}
}

// WithAPIVersion passes the api version to the client. If not set, the default value
// is DefaultAPIVersion.
func WithAPIVersion(apiVersion string) Option {
	return func(opts *options) {
		opts.apiVersion = apiVersion
	}
}

// WithHTTPClient allows setting a custom HTTP client. If not set, the default value
// is http.DefaultClient.
func WithHTTPClient(client openaiclient.Doer) Option {
	return func(opts *options) {
		opts.httpClient = client
	}
}

// WithCallback allows setting a custom Callback Handler.
func WithCallback(callbackHandler callbacks.Handler) Option {
	return func(opts *options) {
		opts.callbackHandler = callbackHandler
	}
}

// WithResponseFormat allows setting a custom response format.
func WithResponseFormat(responseFormat *ResponseFormat) Option {
	return func(opts *options) {
		opts.responseFormat = responseFormat
	}
}

// WithUsingReasoningMaxTokens allows to use reasoning max_tokens instead of effort.
// If reasoning max_tokens is set, it will be sent to the server instead of effort.
// Note: you must use this option within WithModernReasoningFormat(), otherwise it will be ignored.
func WithUsingReasoningMaxTokens() Option {
	return func(opts *options) {
		opts.useReasoningMaxTokens = true
	}
}

// WithModernReasoningFormat includes "reasoning" key and object value in the request payload.
// Otherways, it will be sent as a "reasoning_effort" string value.
func WithModernReasoningFormat() Option {
	return func(opts *options) {
		opts.modernReasoningFormat = true
	}
}

// WithPreserveReasoningContent sends each assistant turn's reasoning back as
// reasoning_content: on the turns that called a tool, and on every assistant
// turn for DeepSeek, Kimi, GLM and Qwen models. A model Mistral serves takes it on
// every turn as a thinking chunk at the head of content instead, and one that
// does not reason there takes none. A MiniMax M-series model on MiniMax's API
// takes it on every turn inside <think> tags at the head of content, unless the
// content already opens with a <think> block. A DeepSeek ("deepseek-") assistant
// turn that holds no reasoning goes back with an empty reasoning_content, which
// DeepSeek's thinking mode requires after the last user message; keep the
// reasoning DeepSeek returned in the history, since the empty field also stops
// DeepSeek from restoring it by the turn's tool call ID.
func WithPreserveReasoningContent() Option {
	return func(opts *options) {
		opts.preserveReasoningContent = true
	}
}

// WithStructuredOutputFallback lets a llms.WithStructuredOutput call reach a
// model whose vendor takes no json_schema response format: DeepSeek
// ("deepseek-" models) and Z.ai GLM ("glm-" models, not the ones Mistral serves),
// which take json_object, and MiniMax M-series models on MiniMax's API, which
// take no response_format at all. Instead of failing with
// [llms.ErrStructuredOutputUnsupported], the call appends the JSON Schema to the
// last user message (or adds a user message when there is none), sends
// response_format json_object where the vendor has it, and reports an
// [llms.WarningSubstitute]. The schema and its name are checked exactly as for
// the native path, so one schema serves both.
//
// Nothing on the server holds the answer to the schema, so it is validated
// locally: a single Markdown code fence around the whole answer and a thinking
// block at its head are removed first (streamed text chunks still carry them),
// and any other answer that is not exactly one JSON value matching the schema
// comes back together with [llms.ErrStructuredOutputValidation], which a caller
// should treat as retryable.
// A tool-call turn and a truncated answer are not validated; on these vendors
// max_tokens also covers the reasoning, so a tight budget can end on "length"
// with an empty answer (see [llms.WithFailOnTruncation]). Every other model keeps
// the native json_schema response format.
func WithStructuredOutputFallback() Option {
	return func(opts *options) {
		opts.structuredOutputFallback = true
	}
}
