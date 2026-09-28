package reasoning

import (
	"fmt"
	"sort"
	"testing"
)

func TestOneModelOneWayToDisableThinking(t *testing.T) {
	t.Parallel()

	snapshot := readMistralListing(t)

	ids := make([]string, 0, len(snapshot))
	for id := range snapshot {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	var split []string
	for _, id := range ids {
		want := ResolveOff(id, ProviderOpenAI)
		for _, alias := range snapshot[id].Aliases {
			if _, known := snapshot[alias]; !known {
				continue
			}
			if got := ResolveOff(alias, ProviderOpenAI); got != want {
				split = append(split, fmt.Sprintf("%s -> %v, but its alias %s -> %v", id, want, alias, got))
			}
		}
	}

	if len(split) > 0 {
		t.Fatalf("the vendor calls these one model, the library disables them differently:\n  %v", split)
	}
}
