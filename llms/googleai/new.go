// package googleai implements a langchaingo provider for Google AI LLMs.
// See https://ai.google.dev/ for more details.
package googleai

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/vxcontrol/langchaingo/callbacks"
	"github.com/vxcontrol/langchaingo/llms"

	"google.golang.org/genai"
)

// GoogleAI is a type that represents a Google AI API client.
type GoogleAI struct {
	CallbacksHandler callbacks.Handler
	client           *genai.Client
	opts             Options
}

var _ llms.Model = &GoogleAI{}

// New creates a new GoogleAI client.
func New(ctx context.Context, opts ...Option) (*GoogleAI, error) {
	clientOptions := DefaultOptions()
	for _, opt := range opts {
		opt(&clientOptions)
	}
	clientOptions.EnsureAuthPresent()

	gi := &GoogleAI{
		opts: clientOptions,
	}

	config := &genai.ClientConfig{}

	// Set HTTP client if provided
	if clientOptions.HTTPClient != nil {
		config.HTTPClient = clientOptions.HTTPClient
	}

	// Determine if we should use Vertex AI or Gemini API
	if clientOptions.CloudProject != "" && clientOptions.CloudLocation != "" {
		if clientOptions.APIKey != "" && !clientOptions.apiKeyFromEnvironment {
			return gi, &ErrOptionNotHonored{
				Options: []string{"WithAPIKey"},
				Reason:  "WithCloudProject and WithCloudLocation select the Vertex AI backend, which takes Google credentials, not an API key",
			}
		}
		config.Backend = genai.BackendVertexAI
		config.Project = clientOptions.CloudProject
		config.Location = clientOptions.CloudLocation
	} else {
		config.Backend = genai.BackendGeminiAPI
		// Use API key from options - EnsureAuthPresent() ensures it's available
		if clientOptions.APIKey != "" {
			config.APIKey = clientOptions.APIKey
		}
		if config.APIKey == "" {
			// Try to get from environment as fallback
			if apiKey := os.Getenv("GOOGLE_API_KEY"); apiKey != "" {
				config.APIKey = apiKey
			}
		}
	}

	if len(clientOptions.ClientOptions) > clientOptions.ownClientOptions {
		clientOptions.unhonoredOnREST = append(clientOptions.unhonoredOnREST, "ClientOptions")
	}
	if len(clientOptions.unhonoredOnREST) > 0 {
		return gi, &ErrOptionNotHonored{Options: clientOptions.unhonoredOnREST}
	}
	if config.Backend == genai.BackendVertexAI || config.APIKey == "" {
		credentials, err := clientOptions.detectCredentials()
		if err != nil {
			return gi, err
		}
		config.Credentials = credentials
	}
	if baseURL := clientOptions.BaseURL; baseURL != "" {
		if !strings.Contains(baseURL, "://") {
			baseURL = "https://" + baseURL
		}
		config.HTTPOptions.BaseURL = baseURL
	}

	client, err := genai.NewClient(ctx, config)
	if err != nil {
		return gi, err
	}

	gi.client = client
	return gi, nil
}

// ErrOptionNotHonored reports options this door cannot carry.
type ErrOptionNotHonored struct {
	Options []string
	Reason  string
}

func (e *ErrOptionNotHonored) Error() string {
	return fmt.Sprintf("googleai: this client cannot honor %s; %s",
		strings.Join(e.Options, ", "), cmp.Or(e.Reason, "shape the transport with WithHTTPClient instead"))
}
