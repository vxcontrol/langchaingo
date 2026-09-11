// Package vertex implements a langchaingo provider for Google Vertex AI LLMs,
// including the Gemini models.
// See https://cloud.google.com/vertex-ai for more details.
package vertex

import (
	"context"
	"errors"
	"os"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/googleai"
)

// ErrMissingCloudTarget reports a client asked for Vertex without naming where.
var ErrMissingCloudTarget = errors.New("vertex: a cloud project and a cloud location are both required")

// Vertex reaches Gemini through the Vertex AI backend of the Google GenAI SDK.
type Vertex struct {
	*googleai.GoogleAI
}

var _ llms.Model = &Vertex{}

// New creates a client bound to the Vertex AI backend. The project and the
// location come from the options, or from GOOGLE_CLOUD_PROJECT and
// GOOGLE_CLOUD_LOCATION when the options leave them empty. Authentication comes
// from googleai.WithCredentialsFile or googleai.WithCredentialsJSON, and from
// application default credentials when neither is given.
func New(ctx context.Context, opts ...googleai.Option) (*Vertex, error) {
	resolved := googleai.DefaultOptions()
	for _, opt := range opts {
		opt(&resolved)
	}

	if resolved.CloudProject == "" {
		if project := os.Getenv("GOOGLE_CLOUD_PROJECT"); project != "" {
			resolved.CloudProject = project
			opts = append(opts, googleai.WithCloudProject(project))
		}
	}
	if resolved.CloudLocation == "" {
		if location := os.Getenv("GOOGLE_CLOUD_LOCATION"); location != "" {
			resolved.CloudLocation = location
			opts = append(opts, googleai.WithCloudLocation(location))
		}
	}
	if resolved.CloudProject == "" || resolved.CloudLocation == "" {
		return nil, ErrMissingCloudTarget
	}

	client, err := googleai.New(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &Vertex{GoogleAI: client}, nil
}

func (v *Vertex) Close() error {
	return nil
}
