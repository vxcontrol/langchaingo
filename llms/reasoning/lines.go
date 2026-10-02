package reasoning

import (
	"cmp"
	"fmt"
	"slices"
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
	_, _, bare := splitModelName(model)
	return inheritLine(bare)
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

type generation struct {
	major, minor int
	members      map[string]string
}

type lineFamily struct {
	prefix     string
	lines      map[string][]generation
	products   []string
	qualifiers []string
	stages     []string
	// dashMinor reads glm-5-3 as 5.3; decimalMinor reads grok-4.20 as 4.2.
	dashMinor, decimalMinor bool
}

var lineFamilies = []lineFamily{
	{
		prefix: "gpt-",
		lines: map[string][]generation{
			"": {
				{5, 0, map[string]string{"": "gpt-5", "mini": "gpt-5-mini", "nano": "gpt-5-nano"}},
				{5, 1, map[string]string{"": "gpt-5.1"}},
				{5, 2, map[string]string{"": "gpt-5.2"}},
				{5, 4, map[string]string{"": "gpt-5.4", "mini": "gpt-5.4-mini", "nano": "gpt-5.4-nano"}},
				{5, 5, map[string]string{"": "gpt-5.5"}},
				{5, 6, map[string]string{"": "gpt-5.6"}},
				{6, 0, map[string]string{"": "gpt-6-sol"}},
				{6, 1, map[string]string{"": "gpt-6.1-sol"}},
			},
			"sol": {
				{5, 6, map[string]string{"": "gpt-5.6-sol"}},
				{6, 0, map[string]string{"": "gpt-6-sol"}},
				{6, 1, map[string]string{"": "gpt-6.1-sol"}},
			},
			"luna":  {{5, 6, map[string]string{"": "gpt-5.6-luna"}}, {6, 0, map[string]string{"": "gpt-6-luna"}}},
			"terra": {{5, 6, map[string]string{"": "gpt-5.6-terra"}}},
			"astra": {{6, 0, map[string]string{"": "gpt-6-astra"}}},
			"cyber": {{5, 6, map[string]string{"": "gpt-5.6-cyber"}}},
			"pro": {
				{5, 0, map[string]string{"": "gpt-5-pro"}}, {5, 2, map[string]string{"": "gpt-5.2-pro"}},
				{5, 4, map[string]string{"": "gpt-5.4-pro"}}, {5, 5, map[string]string{"": "gpt-5.5-pro"}},
			},
		},
		products:   []string{"sol", "luna", "terra", "astra", "cyber", "pro"},
		qualifiers: []string{"mini", "nano"},
	},
	{
		prefix: "glm-",
		lines: map[string][]generation{"": {
			{4, 5, map[string]string{"": "glm-4.5", "air": "glm-4.5-air", "airx": "glm-4.5-airx", "x": "glm-4.5-x", "flash": "glm-4.5-flash"}},
			{4, 6, map[string]string{"": "glm-4.6"}},
			{4, 7, map[string]string{"": "glm-4.7", "flash": "glm-4.7-flash", "flashx": "glm-4.7-flashx"}},
			{5, 0, map[string]string{"": "glm-5"}},
			{5, 1, map[string]string{"": "glm-5.1"}},
			{5, 2, map[string]string{"": "glm-5.2"}},
			{5, 3, map[string]string{"": "glm-5.3", "flash": "glm-5.3-flash", "flashx": "glm-5.3-flashx"}},
		}},
		qualifiers: []string{"air", "airx", "x", "flash", "flashx", "turbo", "fast", "prime"},
		stages:     []string{"preview"},
		dashMinor:  true,
	},
	{
		prefix: "kimi-k",
		lines: map[string][]generation{
			"": {
				{2, 0, map[string]string{"": "kimi-k2"}}, {2, 5, map[string]string{"": "kimi-k2.5"}},
				{2, 6, map[string]string{"": "kimi-k2.6"}}, {3, 0, map[string]string{"": "kimi-k3"}},
			},
			"thinking": {{2, 0, map[string]string{"": "kimi-k2-thinking"}}},
			"code":     {{2, 7, map[string]string{"": "kimi-k2.7-code"}}},
		},
		products:   []string{"thinking", "code"},
		qualifiers: []string{"highspeed"},
		stages:     []string{"preview"},
	},
	{
		prefix: "deepseek-v",
		lines: map[string][]generation{"": {
			{3, 1, map[string]string{"": "deepseek-v3.1"}},
			{3, 2, map[string]string{"": "deepseek-v3.2"}},
			{4, 0, map[string]string{"pro": "deepseek-v4-pro", "flash": "deepseek-v4-flash"}},
			{4, 1, map[string]string{"": "deepseek-flash", "flash": "deepseek-v4.1-flash"}},
		}},
		qualifiers: []string{"pro", "flash"},
		stages:     []string{"preview", "exp"},
	},
	{
		prefix: "minimax-m",
		lines: map[string][]generation{
			"": {
				{2, 0, map[string]string{"": "minimax-m2"}}, {2, 1, map[string]string{"": "minimax-m2.1"}},
				{2, 5, map[string]string{"": "minimax-m2.5"}}, {3, 0, map[string]string{"": "minimax-m3"}},
			},
			"flash": {{3, 1, map[string]string{"": "minimax-m3.1-flash-preview"}}},
		},
		products:   []string{"flash", "her"},
		qualifiers: []string{"highspeed", "stable"},
		stages:     []string{"preview"},
	},
	{
		prefix: "qwen",
		lines: map[string][]generation{"": {
			{3, 0, map[string]string{"max": "qwen3-max"}},
			{3, 5, map[string]string{"plus": "qwen3.5-plus", "flash": "qwen3.5-flash"}},
			{3, 6, map[string]string{"plus": "qwen3.6-plus", "flash": "qwen3.6-flash"}},
			{3, 7, map[string]string{"max": "qwen3.7-max", "plus": "qwen3.7-plus", "flash": "qwen3.7-flash"}},
			{3, 8, map[string]string{"max": "qwen3.8-max", "flash": "qwen3.8-flash"}},
		}},
		products:   []string{"thinking", "instruct", "vl", "omni", "coder", "next"},
		qualifiers: []string{"max", "plus", "flash", "turbo"},
		stages:     []string{"preview"},
	},
	{
		prefix: "grok-",
		lines: map[string][]generation{"": {
			{4, 3, map[string]string{"": "grok-4.3"}}, {4, 5, map[string]string{"": "grok-4.5"}},
			{4, 6, map[string]string{"": "grok-4.6"}}, {4, 7, map[string]string{"": "grok-4.7"}},
		}},
		products:     []string{"reasoning", "non", "multi", "agent", "fast", "code", "build", "mini"},
		decimalMinor: true,
	},
}

type parsedName struct {
	product, qualifier string
	major, minor       int
	dashMinor          bool
	core               string
}

func (f lineFamily) parse(name string) (parsedName, bool) {
	rest, ok := strings.CutPrefix(name, f.prefix)
	if !ok {
		return parsedName{}, false
	}
	tokens := strings.Split(rest, "-")
	version := tokens[0]
	if whole, fraction, dotted := strings.Cut(version, "."); f.decimalMinor && dotted {
		version = whole + "." + cmp.Or(strings.TrimRight(fraction, "0"), "0")
	}
	major, minor, ok := parseVersion(version)
	if !ok {
		return parsedName{}, false
	}
	p := parsedName{major: major, minor: minor}
	core := []string{f.prefix + tokens[0]}
	inDate := false
	for i, token := range tokens[1:] {
		switch {
		case i == 0 && f.dashMinor && !strings.Contains(tokens[0], ".") && len(token) <= 2 && allDigits(token):
			p.minor, _ = strconv.Atoi(token)
			p.dashMinor = true
			core = append(core, token)
		case allDigits(token) && (len(token) >= 4 || inDate):
			inDate = true
		case slices.Contains(f.stages, token):
		case p.product == "" && slices.Contains(f.products, token):
			p.product = token
			core = append(core, token)
		case p.qualifier == "" && slices.Contains(f.qualifiers, token):
			p.qualifier = token
			core = append(core, token)
		default:
			return parsedName{}, false
		}
	}
	p.core = strings.Join(core, "-")
	return p, true
}

func (f lineFamily) inherit(name string) (string, bool) {
	p, ok := f.parse(name)
	if !ok {
		return "", false
	}
	line, ok := f.lines[p.product]
	if !ok {
		return "", false
	}
	var g generation
	found := false
	for _, candidate := range line {
		if candidate.major == p.major && candidate.minor == p.minor {
			g, found = candidate, true
			break
		}
		if (candidate.major < p.major || candidate.major == p.major && candidate.minor < p.minor) &&
			(!found || candidate.major > g.major || candidate.major == g.major && candidate.minor > g.minor) {
			g, found = candidate, true
		}
	}
	if !found {
		return "", false
	}
	documented := g.member(p.qualifier)
	if p.dashMinor {
		documented = strings.Replace(documented, fmt.Sprintf("%d.%d", g.major, g.minor), fmt.Sprintf("%d-%d", g.major, g.minor), 1)
	}
	if documented == name || documented == p.core {
		return "", false
	}
	return documented, true
}

func (g generation) member(qualifier string) string {
	if id, ok := g.members[qualifier]; ok {
		return id
	}
	if id, ok := g.members[""]; ok {
		return id
	}
	for _, preferred := range []string{"max", "pro", "plus", "flash"} {
		if id, ok := g.members[preferred]; ok {
			return id
		}
	}
	return ""
}

func parseVersion(token string) (major, minor int, ok bool) {
	majorText, minorText, dotted := strings.Cut(token, ".")
	if !allDigits(majorText) || dotted && !allDigits(minorText) {
		return 0, 0, false
	}
	major, _ = strconv.Atoi(majorText)
	if dotted {
		minor, _ = strconv.Atoi(minorText)
	}
	return major, minor, true
}

func allDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := range len(s) {
		if !isDigit(s[i]) {
			return false
		}
	}
	return true
}

func inheritLine(bare string) (string, bool) {
	for _, f := range lineFamilies {
		if documented, ok := f.inherit(bare); ok {
			return documented, true
		}
	}
	return "", false
}
