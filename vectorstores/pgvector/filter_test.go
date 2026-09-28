package pgvector

import (
	"errors"
	"strings"
	"testing"
)

func TestFilterPredicatesKeepCallerValuesOutOfTheStatement(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name  string
		value string
	}{
		{"value closes the literal and appends a tautology", "x' OR '1'='1"},
		{"value ends the statement", "x'; DROP TABLE langchaingo_pg_embedding; --"},
		{"quote inside an ordinary value", "O'Brien"},
		{"value carries a comment marker", "x --"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			predicates, args, err := filterPredicates("data.", map[string]any{"kind": tc.value}, 0)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if want := "(data.cmetadata ->> 'kind') = $1"; predicates[0] != want {
				t.Errorf("want %q, got %q", want, predicates[0])
			}
			if len(args) != 1 || args[0] != tc.value {
				t.Errorf("value must travel as an argument, got %v", args)
			}
		})
	}
}

// The key reaches the statement as text, so the gate is the whole defence.
func TestFilterPredicatesRejectAKeyThatIsNotAnIdentifier(t *testing.T) {
	t.Parallel()

	for _, key := range []string{
		"kind') = 'x' OR '1'='1",
		"",
		"1abc",
		"a-b",
		"a.b",
		"a b",
		"a;b",
		`a"b`,
		"a`b",
		"ключ",
		strings.Repeat("a", 64),
	} {
		t.Run(key, func(t *testing.T) {
			t.Parallel()

			predicates, args, err := filterPredicates("data.", map[string]any{key: "y"}, 0)
			if !errors.Is(err, ErrInvalidFilterKey) {
				t.Fatalf("want ErrInvalidFilterKey, got %v", err)
			}
			if predicates != nil || args != nil {
				t.Errorf("a rejected filter must render nothing, got %v and %v", predicates, args)
			}
		})
	}
}

func TestFilterPredicatesNumberFromTheOffset(t *testing.T) {
	t.Parallel()

	predicates, args, err := filterPredicates("data.", map[string]any{"a": "1", "b": "2"}, 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "(data.cmetadata ->> 'a') = $5 AND (data.cmetadata ->> 'b') = $6"
	if got := strings.Join(predicates, " AND "); got != want {
		t.Errorf("want %q, got %q", want, got)
	}
	if len(args) != 2 || args[0] != "1" || args[1] != "2" {
		t.Errorf("want the two values in key order, got %v", args)
	}
}

// Sorting is what keeps one filter rendering as one statement text, which is
// what lets the driver and the server reuse a plan for it.
func TestFilterPredicatesRenderTheSameTextWhateverTheInsertionOrder(t *testing.T) {
	t.Parallel()

	first, _, err := filterPredicates("data.", map[string]any{"flow_id": "1", "doc_type": "memory"}, 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	second, _, err := filterPredicates("data.", map[string]any{"doc_type": "memory", "flow_id": "1"}, 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Join(first, " AND ") != strings.Join(second, " AND ") {
		t.Errorf("statement text varies with map order: %q vs %q", first, second)
	}
}

func TestFilterPredicatesOnValuesThatAreNotStrings(t *testing.T) {
	t.Parallel()

	_, args, err := filterPredicates("t.", map[string]any{"n": 5}, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(args) != 1 || args[0] != "5" {
		t.Errorf("a non-string value must reach the wire as text, got %v", args)
	}
}

func TestFilterPredicatesOnAnEmptyFilter(t *testing.T) {
	t.Parallel()

	predicates, args, err := filterPredicates("data.", map[string]any{}, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(predicates) != 0 || len(args) != 0 {
		t.Errorf("an empty filter must add nothing, got %v and %v", predicates, args)
	}
}
