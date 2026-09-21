package pgvector

import (
	"errors"
	"strings"
	"testing"
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
			want: "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_meta_flow_id " +
				"ON langchain_pg_embedding ((cmetadata ->> 'flow_id'))",
		},
		{
			name:  "keys keep the order they were declared in",
			index: MetadataIndex{Keys: []string{"doc_type", "flow_id"}},
			want: "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_meta_doc_type_flow_id " +
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
			want: "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_meta_doc_type_partial " +
				"ON langchain_pg_embedding ((cmetadata ->> 'doc_type')) " +
				"WHERE (cmetadata ->> 'doc_type') IS DISTINCT FROM 'memory'",
		},
		{
			name: "a quote in an excluded value is doubled",
			index: MetadataIndex{
				Keys:    []string{"owner"},
				Exclude: map[string]string{"owner": "O'Brien"},
			},
			want: "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_meta_owner_partial " +
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

func TestMetadataIndexNameDistinguishesAPartialIndex(t *testing.T) {
	t.Parallel()

	full := MetadataIndex{Keys: []string{"doc_type"}}
	partial := MetadataIndex{Keys: []string{"doc_type"}, Exclude: map[string]string{"doc_type": "memory"}}
	if full.indexName("lpe") == partial.indexName("lpe") {
		t.Error("a partial index must not adopt the full index's name")
	}
}
