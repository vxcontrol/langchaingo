package googleai

import (
	"net/http"
	"os"
	"reflect"

	"cloud.google.com/go/vertexai/genai"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
)

// Options is a set of options for GoogleAI and Vertex clients.
type Options struct {
	CloudProject          string
	CloudLocation         string
	DefaultModel          string
	DefaultEmbeddingModel string
	DefaultCandidateCount int
	DefaultMaxTokens      int
	DefaultTemperature    float64
	DefaultTopK           int
	DefaultTopP           float64
	HarmThreshold         HarmBlockThreshold
	APIKey                string
	HTTPClient            *http.Client

	ClientOptions []option.ClientOption
}

func DefaultOptions() Options {
	return Options{
		CloudProject:          "",
		CloudLocation:         "",
		DefaultModel:          "gemini-2.0-flash",
		DefaultEmbeddingModel: "embedding-001",
		DefaultCandidateCount: 1,
		DefaultMaxTokens:      2048,
		DefaultTemperature:    0.5,
		DefaultTopK:           3,
		DefaultTopP:           0.95,
		HarmThreshold:         HarmBlockNone,
	}
}

// EnsureAuthPresent attempts to ensure that the client has authentication information. If it does not, it will attempt to use the GOOGLE_API_KEY environment variable.
func (o *Options) EnsureAuthPresent() {
	if o.APIKey == "" && !hasAuthOptions(o.ClientOptions) {
		if key := os.Getenv("GOOGLE_API_KEY"); key != "" {
			WithAPIKey(key)(o)
		}
	}
}

type Option func(*Options)

// WithAPIKey passes the API KEY (token) to the client. This is useful for
// googleai clients.
func WithAPIKey(apiKey string) Option {
	return func(opts *Options) {
		opts.APIKey = apiKey
		opts.ClientOptions = append(opts.ClientOptions, option.WithAPIKey(apiKey))
	}
}

// WithCredentialsJSON append a ClientOption that authenticates
// API calls with the given service account or refresh token JSON
// credentials.
func WithCredentialsJSON(credentialsJSON []byte) Option {
	return func(opts *Options) {
		if len(credentialsJSON) == 0 {
			return
		}
		opts.ClientOptions = append(opts.ClientOptions, option.WithCredentialsJSON(credentialsJSON))
	}
}

// WithCredentialsFile append a ClientOption that authenticates
// API calls with the given service account or refresh token JSON
// credentials file.
func WithCredentialsFile(credentialsFile string) Option {
	return func(opts *Options) {
		if credentialsFile == "" {
			return
		}
		opts.ClientOptions = append(opts.ClientOptions, option.WithCredentialsFile(credentialsFile))
	}
}

// WithRest configures the client to use the REST API.
func WithRest() Option {
	return func(opts *Options) {
		opts.ClientOptions = append(opts.ClientOptions, genai.WithREST())
	}
}

// WithGRPCClient append a ClientOption that uses the provided gRPC client to
// make requests.
// This is useful for gemini clients.
func WithGRPCClient(grpcClient *grpc.ClientConn) Option {
	return func(opts *Options) {
		opts.ClientOptions = append(opts.ClientOptions, option.WithGRPCConn(grpcClient))
	}
}

// WithEndpoint append a ClientOption that uses the provided endpoint to
// make requests.
// This is useful for gemini clients.
func WithEndpoint(endpoint string) Option {
	return func(opts *Options) {
		opts.ClientOptions = append(opts.ClientOptions, option.WithEndpoint(endpoint))
	}
}

// WithHTTPClient append a ClientOption that uses the provided HTTP client to
// make requests.
// This is useful for vertex clients.
func WithHTTPClient(httpClient *http.Client) Option {
	return func(opts *Options) {
		opts.HTTPClient = httpClient
		opts.ClientOptions = append(opts.ClientOptions, option.WithHTTPClient(httpClient))
	}
}

// WithCloudProject passes the GCP cloud project name to the client. This is
// useful for vertex clients.
func WithCloudProject(p string) Option {
	return func(opts *Options) {
		opts.CloudProject = p
	}
}

// WithCloudLocation passes the GCP cloud location (region) name to the client.
// This is useful for vertex clients.
func WithCloudLocation(l string) Option {
	return func(opts *Options) {
		opts.CloudLocation = l
	}
}

// WithDefaultModel passes a default content model name to the client. This
// model name is used if not explicitly provided in specific client invocations.
func WithDefaultModel(defaultModel string) Option {
	return func(opts *Options) {
		opts.DefaultModel = defaultModel
	}
}

// WithDefaultModel passes a default embedding model name to the client. This
// model name is used if not explicitly provided in specific client invocations.
func WithDefaultEmbeddingModel(defaultEmbeddingModel string) Option {
	return func(opts *Options) {
		opts.DefaultEmbeddingModel = defaultEmbeddingModel
	}
}

// WithDefaultCandidateCount sets the candidate count for the model.
func WithDefaultCandidateCount(defaultCandidateCount int) Option {
	return func(opts *Options) {
		opts.DefaultCandidateCount = defaultCandidateCount
	}
}

// WithDefaultMaxTokens sets the maximum token count for the model.
func WithDefaultMaxTokens(maxTokens int) Option {
	return func(opts *Options) {
		opts.DefaultMaxTokens = maxTokens
	}
}

// WithDefaultTemperature sets the maximum token count for the model.
func WithDefaultTemperature(defaultTemperature float64) Option {
	return func(opts *Options) {
		opts.DefaultTemperature = defaultTemperature
	}
}

// WithDefaultTopK sets the TopK for the model.
func WithDefaultTopK(defaultTopK int) Option {
	return func(opts *Options) {
		opts.DefaultTopK = defaultTopK
	}
}

// WithDefaultTopP sets the TopP for the model.
func WithDefaultTopP(defaultTopP float64) Option {
	return func(opts *Options) {
		opts.DefaultTopP = defaultTopP
	}
}

// WithHarmThreshold sets the safety/harm setting for the model, potentially
// limiting any harmful content it may generate.
func WithHarmThreshold(ht HarmBlockThreshold) Option {
	return func(opts *Options) {
		opts.HarmThreshold = ht
	}
}

type HarmBlockThreshold string

const (
	// HarmBlockUnspecified means threshold is unspecified.
	HarmBlockUnspecified HarmBlockThreshold = "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
	// HarmBlockLowAndAbove means content with NEGLIGIBLE will be allowed.
	HarmBlockLowAndAbove HarmBlockThreshold = "BLOCK_LOW_AND_ABOVE"
	// HarmBlockMediumAndAbove means content with NEGLIGIBLE and LOW will be allowed.
	HarmBlockMediumAndAbove HarmBlockThreshold = "BLOCK_MEDIUM_AND_ABOVE"
	// HarmBlockOnlyHigh means content with NEGLIGIBLE, LOW, and MEDIUM will be allowed.
	HarmBlockOnlyHigh HarmBlockThreshold = "BLOCK_ONLY_HIGH"
	// HarmBlockNone means all content will be allowed.
	HarmBlockNone HarmBlockThreshold = "BLOCK_NONE"
)

// helper to inspect incoming client options for auth options.
func hasAuthOptions(opts []option.ClientOption) bool {
	for _, opt := range opts {
		v := reflect.ValueOf(opt)
		ts := v.Type().String()

		switch ts {
		case "option.withAPIKey":
			return v.String() != ""

		case "option.withTokenSource",
			"option.withCredentialsFile",
			"option.withCredentialsJSON":
			return true
		}
	}
	return false
}
