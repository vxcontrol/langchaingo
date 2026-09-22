package pgvector

import (
	"errors"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestMetadataIndexDDL(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name  string
		index MetadataIndex
		want  string
	}{
		{
			name:  "one key",
			index: MetadataIndex{Keys: []string{"flow_id"}},
			want: "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_meta_flow_id_6de4f5f1 " +
				"ON langchain_pg_embedding ((cmetadata ->> 'flow_id'))",
		},
		{
			name:  "keys keep the order they were declared in",
			index: MetadataIndex{Keys: []string{"doc_type", "flow_id"}},
			want: "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_meta_doc_type_flow_id_6c451f01 " +
				"ON langchain_pg_embedding ((cmetadata ->> 'doc_type'), (cmetadata ->> 'flow_id'))",
		},
		{
			name:  "an explicit name wins",
			index: MetadataIndex{Name: "lpe_custom", Keys: []string{"flow_id"}},
			want:  "CREATE INDEX IF NOT EXISTS lpe_custom ON langchain_pg_embedding ((cmetadata ->> 'flow_id'))",
		},
		{
			name: "exclude renders a partial index",
			index: MetadataIndex{
				Keys:    []string{"doc_type"},
				Exclude: map[string]string{"doc_type": "memory"},
			},
			want: "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_meta_doc_type_partial_bf975523 " +
				"ON langchain_pg_embedding ((cmetadata ->> 'doc_type')) " +
				"WHERE (cmetadata ->> 'doc_type') IS DISTINCT FROM 'memory'",
		},
		{
			name: "a quote in an excluded value is doubled",
			index: MetadataIndex{
				Keys:    []string{"owner"},
				Exclude: map[string]string{"owner": "O'Brien"},
			},
			want: "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_meta_owner_partial_fb6d36ce " +
				"ON langchain_pg_embedding ((cmetadata ->> 'owner')) " +
				"WHERE (cmetadata ->> 'owner') IS DISTINCT FROM 'O''Brien'",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := tc.index.ddl("langchain_pg_embedding")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("want:\n%s\ngot:\n%s", tc.want, got)
			}
		})
	}
}

func TestMetadataIndexRejectsWhatCannotBeInlinedSafely(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name  string
		index MetadataIndex
	}{
		{"no keys", MetadataIndex{}},
		{"key is not an identifier", MetadataIndex{Keys: []string{"a-b"}}},
		{"key closes the literal", MetadataIndex{Keys: []string{"x') OR true --"}}},
		{"excluded key is not an identifier", MetadataIndex{
			Keys: []string{"doc_type"}, Exclude: map[string]string{"a.b": "x"},
		}},
		{"name is not an identifier", MetadataIndex{Name: "a b", Keys: []string{"doc_type"}}},
		{"excluded value carries a backslash", MetadataIndex{
			Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": `a\b`},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if _, err := tc.index.ddl("langchain_pg_embedding"); !errors.Is(err, ErrInvalidMetadataIndex) {
				t.Fatalf("want ErrInvalidMetadataIndex, got %v", err)
			}
		})
	}
}

// PostgreSQL truncates an identifier at 63 bytes, which would collide two
// distinct declarations under one index.
func TestMetadataIndexNameStaysWithinTheIdentifierLimit(t *testing.T) {
	t.Parallel()

	long := MetadataIndex{Keys: []string{
		strings.Repeat("a", 60), strings.Repeat("b", 60), strings.Repeat("c", 60),
	}}
	first := long.indexName("langchain_pg_embedding")
	if len(first) > maxIdentifierLen {
		t.Fatalf("name is %d bytes: %s", len(first), first)
	}

	other := MetadataIndex{Keys: []string{
		strings.Repeat("a", 60), strings.Repeat("b", 60), strings.Repeat("d", 60),
	}}
	if second := other.indexName("langchain_pg_embedding"); first == second {
		t.Errorf("two declarations collapsed onto one name: %s", first)
	}
}

func TestMetadataIndexNameCutsATableNameOnACharacterBoundary(t *testing.T) {
	t.Parallel()

	name := MetadataIndex{Keys: []string{"k"}}.indexName("t" + strings.Repeat("ж", 27))
	if !utf8.ValidString(name) || len(name) > maxIdentifierLen {
		t.Errorf("name is %d bytes, valid UTF-8 = %v: %q", len(name), utf8.ValidString(name), name)
	}
}

func TestMetadataIndexNameDistinguishesEveryDeclaration(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		a, b MetadataIndex
	}{
		{
			"a partial index and the full one",
			MetadataIndex{Keys: []string{"doc_type"}},
			MetadataIndex{Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "memory"}},
		},
		{
			"two partial indexes excluding different values",
			MetadataIndex{Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "memory"}},
			MetadataIndex{Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "session"}},
		},
		{
			"keys that regroup around an underscore",
			MetadataIndex{Keys: []string{"a_b", "c"}},
			MetadataIndex{Keys: []string{"a", "b_c"}},
		},
		{
			"a key named partial and a partial index",
			MetadataIndex{Keys: []string{"flow_id", "partial"}},
			MetadataIndex{Keys: []string{"flow_id"}, Exclude: map[string]string{"doc_type": "memory"}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if name := tc.a.indexName("lpe"); name == tc.b.indexName("lpe") {
				t.Errorf("two declarations collapsed onto one name: %s", name)
			}
		})
	}
}

func TestMetadataIndexesSharingANameMustAgree(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name    string
		indexes []MetadataIndex
		refused bool
	}{
		{"different keys under one name", []MetadataIndex{
			{Name: "lpe_shared", Keys: []string{"flow_id"}},
			{Name: "lpe_shared", Keys: []string{"doc_type"}},
		}, true},
		{"names that differ only in case", []MetadataIndex{
			{Name: "Lpe_shared", Keys: []string{"flow_id"}},
			{Name: "lpe_shared", Keys: []string{"doc_type"}},
		}, true},
		{"exclusions that differ under one name", []MetadataIndex{
			{Name: "lpe_shared", Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "memory"}},
			{Name: "lpe_shared", Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "session"}},
		}, true},
		{"names that differ only in case over the same keys", []MetadataIndex{
			{Name: "Lpe_shared", Keys: []string{"flow_id"}},
			{Name: "lpe_shared", Keys: []string{"flow_id"}},
		}, false},
		{"the same declaration twice", []MetadataIndex{
			{Name: "lpe_shared", Keys: []string{"flow_id"}},
			{Name: "lpe_shared", Keys: []string{"flow_id"}},
		}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			store := Store{embeddingTableName: "langchain_pg_embedding", metadataIndexes: tc.indexes}
			err := store.createMetadataIndexesIfNotExist(t.Context(), &scriptedTx{})
			if refused := errors.Is(err, ErrInvalidMetadataIndex); refused != tc.refused || (!refused && err != nil) {
				t.Fatalf("refused = %v, want %v (err: %v)", refused, tc.refused, err)
			}
		})
	}
}
