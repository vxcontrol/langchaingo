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

// ErrMissingCloudTarget reports a client asked for Vertex without naming a project.
var ErrMissingCloudTarget = errors.New("vertex: a cloud project is required")

var (
	ErrNoContentInResponse   = googleai.ErrNoContentInResponse
	ErrUnknownPartInResponse = googleai.ErrUnknownPartInResponse
	ErrInvalidMimeType       = googleai.ErrInvalidMimeType
)

const (
	CITATIONS            = googleai.CITATIONS
	SAFETY               = googleai.SAFETY
	RoleSystem           = googleai.RoleSystem
	RoleModel            = googleai.RoleModel
	RoleUser             = googleai.RoleUser
	RoleTool             = googleai.RoleTool
	ResponseMIMETypeJson = googleai.ResponseMIMETypeJson
)

const defaultLocation = "us-central1"

// Vertex reaches Gemini through the Vertex AI backend of the Google GenAI SDK.
type Vertex struct {
	*googleai.GoogleAI
}

var _ llms.Model = &Vertex{}

// New creates a client bound to the Vertex AI backend. The project comes from
// the options or GOOGLE_CLOUD_PROJECT; the location from the options,
// GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_REGION or CLOUD_ML_REGION, else
// us-central1. Authentication comes from googleai.WithCredentialsFile or
// googleai.WithCredentialsJSON, and from application default credentials when
// neither is given.
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
	if resolved.CloudProject == "" {
		return nil, ErrMissingCloudTarget
	}
	if resolved.CloudLocation == "" {
		opts = append(opts, googleai.WithCloudLocation(locationFromEnvironment()))
	}

	client, err := googleai.New(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &Vertex{GoogleAI: client}, nil
}

func locationFromEnvironment() string {
	for _, name := range []string{"GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION", "CLOUD_ML_REGION"} {
		if location := os.Getenv(name); location != "" {
			return location
		}
	}
	return defaultLocation
}

func (v *Vertex) Close() error {
	return nil
}
