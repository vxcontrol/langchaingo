package ollama

import (
	"log"
	"net/http"
	"net/url"
	"time"

	"github.com/ollama/ollama/api"
)

type options struct {
	ollamaServerURL     *url.URL
	httpClient          *http.Client
	model               string
	ollamaOptions       api.Options
	customModelTemplate string
	system              string
	format              string
	keepAlive           *time.Duration
	pullModel           bool
	pullTimeout         time.Duration
}

type Option func(*options)

// WithModel Set the model to use.
func WithModel(model string) Option {
	return func(opts *options) {
		opts.model = model
	}
}

// WithFormat Sets the Ollama output format (currently Ollama only supports "json").
func WithFormat(format string) Option {
	return func(opts *options) {
		opts.format = format
	}
}

// WithKeepAlive controls how long the model will stay loaded into memory following the request (default: 5m)
// only supported by ollama v0.1.23 and later
//
//	If set to a positive duration (e.g. 20m, 1h or 30), the model will stay loaded for the provided duration
//	If set to a negative duration (e.g. -1), the model will stay loaded indefinitely
//	If set to 0, the model will be unloaded immediately once finished
//	If not set, the model will stay loaded for 5 minutes by default
func WithKeepAlive(keepAlive string) Option {
	return func(opts *options) {
		ka, err := time.ParseDuration(keepAlive)
		if err != nil {
			log.Fatal(err)
		}
		opts.keepAlive = &ka
	}
}

// WithSystemPrompt Set the system prompt. This is only valid if
// WithCustomTemplate is not set and the ollama model use
// .System in its model template OR if WithCustomTemplate
// is set using {{.System}}.
func WithSystemPrompt(p string) Option {
	return func(opts *options) {
		opts.system = p
	}
}

// WithCustomTemplate To override the templating done on Ollama model side.
func WithCustomTemplate(template string) Option {
	return func(opts *options) {
		opts.customModelTemplate = template
	}
}

// WithServerURL Set the URL of the ollama instance to use.
func WithServerURL(rawURL string) Option {
	return func(opts *options) {
		var err error
		opts.ollamaServerURL, err = url.Parse(rawURL)
		if err != nil {
			log.Fatal(err)
		}
	}
}

// WithHTTPClient Set custom http client.
func WithHTTPClient(client *http.Client) Option {
	return func(opts *options) {
		opts.httpClient = client
	}
}

// WithRunnerNumCtx Sets the size of the context window used to generate the next token (Default: 2048).
func WithRunnerNumCtx(num int) Option {
	return func(opts *options) {
		opts.ollamaOptions.NumCtx = num
	}
}

// WithRunnerNumKeep Specify the number of tokens from the initial prompt to retain when the model resets
// its internal context.
func WithRunnerNumKeep(num int) Option {
	return func(opts *options) {
		opts.ollamaOptions.NumKeep = num
	}
}

// WithRunnerNumBatch Set the batch size for prompt processing (default: 512).
func WithRunnerNumBatch(num int) Option {
	return func(opts *options) {
		opts.ollamaOptions.NumBatch = num
	}
}

// WithRunnerNumThread Set the number of threads to use during computation (default: auto).
func WithRunnerNumThread(num int) Option {
	return func(opts *options) {
		opts.ollamaOptions.NumThread = num
	}
}

// WithRunnerNumGPU The number of layers to send to the GPU(s).
// On macOS it defaults to 1 to enable metal support, 0 to disable.
func WithRunnerNumGPU(num int) Option {
	return func(opts *options) {
		opts.ollamaOptions.NumGPU = num
	}
}

// WithRunnerMainGPU When using multiple GPUs this option controls which GPU is used for small tensors
// for which the overhead of splitting the computation across all GPUs is not worthwhile.
// The GPU in question will use slightly more VRAM to store a scratch buffer for temporary results.
// By default GPU 0 is used.
func WithRunnerMainGPU(num int) Option {
	return func(opts *options) {
		opts.ollamaOptions.MainGPU = num
	}
}

// WithRunnerUseMMap Set to false to not memory-map the model.
// By default, models are mapped into memory, which allows the system to load only the necessary parts
// of the model as needed.
func WithRunnerUseMMap(val bool) Option {
	return func(opts *options) {
		opts.ollamaOptions.UseMMap = &val
	}
}

// WithPredictRepeatLastN Sets how far back for the model to look back to prevent repetition
// (Default: 64, 0 = disabled, -1 = num_ctx).
func WithPredictRepeatLastN(val int) Option {
	return func(opts *options) {
		opts.ollamaOptions.RepeatLastN = val
	}
}

// WithPullModel enables automatic model pulling before use.
// When enabled, the client will check if the model exists and pull it if not available.
func WithPullModel() Option {
	return func(opts *options) {
		opts.pullModel = true
	}
}

// WithPullTimeout sets a timeout for model pulling operations.
// If not set or if duration is 0, pull operations will use the request context without additional timeout.
// This option only takes effect when WithPullModel is also enabled.
func WithPullTimeout(timeout time.Duration) Option {
	return func(opts *options) {
		opts.pullTimeout = timeout
	}
}
