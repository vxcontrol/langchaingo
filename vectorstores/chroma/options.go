package chroma

import (
	"errors"
	"fmt"
	"os"

	"github.com/vxcontrol/langchaingo/embeddings"

	chromatypes "github.com/amikos-tech/chroma-go/types"
)

const (
	ChromaURLKeyEnvVarName = "CHROMA_URL"
	DefaultNameSpace       = "langchain"
	DefaultNameSpaceKey    = "nameSpace"
	DefaultDistanceFunc    = chromatypes.L2
)

// ErrInvalidOptions is returned when the options given are invalid.
var ErrInvalidOptions = errors.New("invalid options")

// Option is a function type that can be used to modify the client.
type Option func(p *Store)

// WithNameSpace sets the nameSpace used to upsert and query the vectors from.
func WithNameSpace(nameSpace string) Option {
	return func(p *Store) {
		p.nameSpace = nameSpace
	}
}

// WithChromaURL is an option for specifying the Chroma URL. Must be set.
func WithChromaURL(chromaURL string) Option {
	return func(p *Store) {
		p.chromaURL = chromaURL
	}
}

// WithEmbedder is an option for setting the embedder to use.
func WithEmbedder(e embeddings.Embedder) Option {
	return func(p *Store) {
		p.embedder = e
	}
}

// WithDistanceFunction specifies the distance function which will be used (default is L2)
// see: https://github.com/amikos-tech/chroma-go/blob/ab1339d0ee1a863be7d6773bcdedc1cfd08e3d77/types/types.go#L22
func WithDistanceFunction(distanceFunction chromatypes.DistanceFunction) Option {
	return func(p *Store) {
		p.distanceFunction = distanceFunction
	}
}

// WithIncludes is an option for setting the includes to query the vectors.
func WithIncludes(includes []chromatypes.QueryEnum) Option {
	return func(p *Store) {
		p.includes = includes
	}
}

func applyClientOptions(opts ...Option) (Store, error) {
	o := &Store{
		nameSpace:        DefaultNameSpace,
		nameSpaceKey:     DefaultNameSpaceKey,
		distanceFunction: DefaultDistanceFunc,
	}

	for _, opt := range opts {
		opt(o)
	}

	if o.chromaURL == "" {
		o.chromaURL = os.Getenv(ChromaURLKeyEnvVarName)
		if o.chromaURL == "" {
			return Store{}, fmt.Errorf(
				"%w: missing chroma URL. Pass it as an option or set the %s environment variable",
				ErrInvalidOptions, ChromaURLKeyEnvVarName)
		}
	}

	return *o, nil
}
