package reasoning

import (
	"strconv"
	"strings"
)

type release struct {
	major, minor int
	id           string
}

func (r release) atOrBelow(major, minor int) bool {
	return r.major < major || r.major == major && r.minor <= minor
}

func nearestRelease(releases []release, major, minor int) (nearest release, documented, found bool) {
	for _, r := range releases {
		if r.major == major && r.minor == minor {
			return r, true, true
		}
		if r.atOrBelow(major, minor) && (!found || !r.atOrBelow(nearest.major, nearest.minor)) {
			nearest, found = r, true
		}
	}
	return nearest, false, found
}

// InheritedModel returns the documented model whose rules a name follows when the
// tables do not list the name's own version: claude-opus-6 follows claude-opus-5-5.
func InheritedModel(model string) (string, bool) {
	m := claudeName(model)
	if idx := strings.Index(m, "claude-"); idx != -1 {
		return inheritClaude(m[idx:])
	}
	return "", false
}

var claudeReleases = map[string][]release{
	"opus": {
		{4, 0, "claude-opus-4-0"}, {4, 1, "claude-opus-4-1"}, {4, 5, "claude-opus-4-5"},
		{4, 6, "claude-opus-4-6"}, {4, 7, "claude-opus-4-7"}, {4, 8, "claude-opus-4-8"},
		{5, 0, "claude-opus-5"}, {5, 5, "claude-opus-5-5"},
	},
	"sonnet": {
		{4, 0, "claude-sonnet-4-0"}, {4, 5, "claude-sonnet-4-5"}, {4, 6, "claude-sonnet-4-6"},
		{5, 0, "claude-sonnet-5"}, {5, 5, "claude-sonnet-5-5"},
	},
	"haiku":  {{4, 5, "claude-haiku-4-5"}},
	"fable":  {{5, 0, "claude-fable-5"}, {5, 1, "claude-fable-5-1"}},
	"mythos": {{5, 0, "claude-mythos-5"}, {5, 1, "claude-mythos-5-1"}},
}

func inheritClaude(canonical string) (string, bool) {
	rest, ok := strings.CutPrefix(canonical, "claude-")
	if !ok {
		return "", false
	}
	parts := strings.Split(rest, "-")
	releases, known := claudeReleases[parts[0]]
	if !known || len(parts) < 2 {
		return "", false
	}
	major, err := strconv.Atoi(parts[1])
	if err != nil {
		return "", false
	}
	minor := 0
	if len(parts) > 2 && len(parts[2]) <= 2 {
		if n, err := strconv.Atoi(parts[2]); err == nil {
			minor = n
		}
	}
	r, documented, found := nearestRelease(releases, major, minor)
	if documented || !found {
		return "", false
	}
	return r.id, true
}
