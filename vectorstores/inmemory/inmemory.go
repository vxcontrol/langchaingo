package inmemory

import (
	"context"
	"errors"
	"slices"
	"strconv"
	"sync"

	"github.com/coder/hnsw"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

var (
	ErrEmbedderWrongNumberVectors = errors.New("number of vectors from embedder does not match number of documents")
	ErrInvalidScoreThreshold      = errors.New("score threshold must be between 0 and 1")
	ErrUnsupportedOptions         = errors.New("unsupported options")
)

// Store is a struct that holds the in-memory vector store.
type Store struct {
	sync.RWMutex

	// HNSW index
	index      *hnsw.Graph[uint32]
	vectorSize int
	content    map[uint32]string
	meta       map[uint32]map[string]any
	embedder   embeddings.Embedder

	// HNSW index parameters
	m              int
	efConstruction int
	efSearch       int

	// size limit of the store
	sizeLimit int
	lastID    uint32
}

// New returns a new InMemory store with options.
func New(ctx context.Context, opts ...Option) (*Store, error) {
	// Currently, we don't use the context.
	// But adding it for API consistency with other vectorstores.
	_ = ctx

	store := applyOptions(opts)

	// Initialize the HNSW graph
	store.index = hnsw.NewGraph[uint32]()

	// Configure graph parameters
	store.index.M = store.m
	store.index.Ml = 0.5 // Default parameter in the library

	// Set the distance function to cosine distance
	store.index.Distance = hnsw.CosineDistance

	// Set efSearch parameter
	store.index.EfSearch = store.efSearch

	// Initialize maps
	store.content = make(map[uint32]string)
	store.meta = make(map[uint32]map[string]any)

	return store, nil
}

// AddDocuments adds documents to the in-memory store
// and returns the ids of the added documents.
func (s *Store) AddDocuments(
	ctx context.Context,
	docs []schema.Document,
	options ...vectorstores.Option,
) ([]string, error) {
	opts := s.getOptions(options...)
	if opts.NameSpace != "" {
		// in-memory store does not support these options
		return nil, ErrUnsupportedOptions
	}

	docs = s.deduplicate(ctx, opts, docs)

	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		texts = append(texts, doc.PageContent)
	}

	embedder := s.embedder
	if opts.Embedder != nil {
		embedder = opts.Embedder
	}

	vectors, err := embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return nil, err
	}

	if len(vectors) != len(docs) {
		return nil, ErrEmbedderWrongNumberVectors
	}

	ids := make([]string, len(vectors))
	for i, vec := range vectors {
		s.Lock()

		id := s.lastID + 1

		s.index.Add(hnsw.MakeNode(id, vec))
		s.lastID = id

		s.content[id] = docs[i].PageContent
		s.meta[id] = docs[i].Metadata

		s.Unlock()

		ids[i] = strconv.FormatUint(uint64(id), 10)
	}

	return ids, nil
}

func (s *Store) SimilaritySearch(
	ctx context.Context,
	query string,
	numDocuments int,
	options ...vectorstores.Option,
) ([]schema.Document, error) {
	opts := s.getOptions(options...)
	if opts.NameSpace != "" {
		// in-memory store does not support these options
		return nil, ErrUnsupportedOptions
	}

	var filters map[string]any
	if f, ok := opts.Filters.(map[string]any); ok {
		filters = f
	}

	embedder := s.embedder
	if opts.Embedder != nil {
		embedder = opts.Embedder
	}
	embedderData, err := embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}

	// this returns a slice of Node objects that contain both the key and the vector
	neighbors := s.index.Search(embedderData, numDocuments)

	docs := make([]schema.Document, 0, len(neighbors))
	for _, n := range neighbors {
		s.RLock()

		// calculate similarity score as 1 - cosine distance
		similarity := 1.0 - float64(hnsw.CosineDistance(embedderData, n.Value))

		doc := schema.Document{
			PageContent: s.content[n.Key],
			Metadata:    s.meta[n.Key],
			Score:       float32(similarity),
		}
		s.RUnlock()

		docs = append(docs, doc)
	}

	docs = applyFilters(docs, filters)
	docs, err = applyScoreThreshold(docs, opts.ScoreThreshold)
	if err != nil {
		return nil, err
	}

	// already sorted by increasing distance, so reverse to get highest similarity first
	slices.Reverse(docs)
	return docs, nil
}

// getOptions applies given options to default Options and returns it
// This uses options pattern so clients can easily pass options without changing function signature.
func (s *Store) getOptions(options ...vectorstores.Option) vectorstores.Options {
	opts := vectorstores.Options{}
	for _, opt := range options {
		opt(&opts)
	}
	return opts
}

// deduplicate applies the deduplicater to the given slice of documents.
// It returns a new slice of documents with the duplicates removed.
func (s *Store) deduplicate(
	ctx context.Context,
	opts vectorstores.Options,
	docs []schema.Document,
) []schema.Document {
	if opts.Deduplicater == nil {
		return docs
	}

	filtered := make([]schema.Document, 0, len(docs))
	for _, doc := range docs {
		if !opts.Deduplicater(ctx, doc) {
			filtered = append(filtered, doc)
		}
	}

	return filtered
}

// applyScoreThreshold applies the score threshold to the given slice of documents.
func applyScoreThreshold(docs []schema.Document, threshold float32) ([]schema.Document, error) {
	if threshold < 0 || threshold > 1 {
		return nil, ErrInvalidScoreThreshold
	}

	filtered := make([]schema.Document, 0, len(docs))
	for _, doc := range docs {
		if doc.Score >= threshold {
			filtered = append(filtered, doc)
		}
	}
	return filtered, nil
}

// applyFilters applies the filters to the given slice of documents.
func applyFilters(docs []schema.Document, filters map[string]any) []schema.Document {
	filtered := make([]schema.Document, 0, len(docs))
	for _, doc := range docs {
		if matchesFilters(doc.Metadata, filters) {
			filtered = append(filtered, doc)
		}
	}
	return filtered
}

// matchesFilters returns true if the given metadata matches the filters.
func matchesFilters(meta map[string]any, filters map[string]any) bool {
	for k, v := range filters {
		if meta[k] != v {
			return false
		}
	}
	return true
}
