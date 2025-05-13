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

	// required when APIType is APITypeAzure or APITypeAzureAD
	apiVersion     string
	embeddingModel string

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
// is read from the OPENAI_API_KEY environment variable.
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
