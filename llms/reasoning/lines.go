package reasoning

import (
	"cmp"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

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

var claudeReleases = map[string][]generation{
	"opus": {
		only(4, 0, "claude-opus-4-0"), only(4, 1, "claude-opus-4-1"), only(4, 5, "claude-opus-4-5"),
		only(4, 6, "claude-opus-4-6"), only(4, 7, "claude-opus-4-7"), only(4, 8, "claude-opus-4-8"),
		only(5, 0, "claude-opus-5"), only(5, 5, "claude-opus-5-5"),
	},
	"sonnet": {
		only(4, 0, "claude-sonnet-4-0"), only(4, 5, "claude-sonnet-4-5"), only(4, 6, "claude-sonnet-4-6"),
		only(5, 0, "claude-sonnet-5"), only(5, 5, "claude-sonnet-5-5"),
	},
	"haiku":  {only(4, 5, "claude-haiku-4-5")},
	"fable":  {only(5, 0, "claude-fable-5"), only(5, 1, "claude-fable-5-1")},
	"mythos": {only(5, 0, "claude-mythos-5"), only(5, 1, "claude-mythos-5-1")},
}

func claudeVersion(canonical string) (tier string, major, minor int, ok bool) {
	rest, ok := strings.CutPrefix(canonical, "claude-")
	if !ok {
		return "", 0, 0, false
	}
	parts := strings.Split(rest, "-")
	if len(parts) < 2 {
		return "", 0, 0, false
	}
	major, err := strconv.Atoi(parts[1])
	if err != nil {
		return "", 0, 0, false
	}
	if len(parts) > 2 && len(parts[2]) <= 2 {
		if n, err := strconv.Atoi(parts[2]); err == nil {
			minor = n
		}
	}
	return parts[0], major, minor, true
}

func inheritClaude(canonical string) (string, bool) {
	tier, major, minor, ok := claudeVersion(canonical)
	if !ok {
		return "", false
	}
	g, listed, found := nearest(claudeReleases[tier], major, minor)
	if listed || !found {
		return "", false
	}
	return g.members[""], true
}

type generation struct {
	major, minor int
	members      map[string]string
}

func only(major, minor int, id string) generation {
	return generation{major, minor, map[string]string{"": id}}
}

func (g generation) before(major, minor int) bool {
	return g.major < major || g.major == major && g.minor < minor
}

func nearest(line []generation, major, minor int) (g generation, listed, found bool) {
	for _, candidate := range line {
		if candidate.major == major && candidate.minor == minor {
			return candidate, true, true
		}
		if candidate.before(major, minor) && (!found || g.before(candidate.major, candidate.minor)) {
			g, found = candidate, true
		}
	}
	return g, false, found
}

func atOrBelow(line []generation, top generation) []generation {
	var below []generation
	for _, g := range line {
		if !top.before(g.major, g.minor) {
			below = append(below, g)
		}
	}
	slices.SortFunc(below, func(a, b generation) int {
		return cmp.Or(cmp.Compare(b.major, a.major), cmp.Compare(b.minor, a.minor))
	})
	return below
}

type lineFamily struct {
	prefix     string
	lines      map[string][]generation
	products   []string
	qualifiers []string
	stages     []string
	// compounds maps a token to the product it extends: lite after flash is flash-lite.
	compounds map[string]string
	// sizes lets parameter counts (26b, a4b) stand without changing the line.
	sizes bool
	// dashMinor reads glm-5-3 as 5.3; decimalMinor reads grok-4.20 as 4.2.
	dashMinor, decimalMinor bool
}

var lineFamilies = []lineFamily{
	{
		prefix: "gemini-",
		lines: map[string][]generation{
			"flash": {
				{2, 0, map[string]string{"": "gemini-2.0-flash"}}, {2, 5, map[string]string{"": "gemini-2.5-flash"}},
				{3, 0, map[string]string{"": "gemini-3-flash-preview"}}, {3, 5, map[string]string{"": "gemini-3.5-flash"}},
				{3, 6, map[string]string{"": "gemini-3.6-flash"}}, {3, 7, map[string]string{"": "gemini-3.7-flash"}},
				{3, 8, map[string]string{"": "gemini-3.8-flash"}},
			},
			"pro": {
				{2, 5, map[string]string{"": "gemini-2.5-pro"}}, {3, 0, map[string]string{"": "gemini-3-pro-preview"}},
				{3, 1, map[string]string{
					"": "gemini-3.1-pro-preview", "customtools": "gemini-3.1-pro-preview-customtools",
				}},
			},
			"flash-lite": {
				{2, 5, map[string]string{"": "gemini-2.5-flash-lite"}},
				{3, 1, map[string]string{"": "gemini-3.1-flash-lite-preview"}},
				{3, 5, map[string]string{"": "gemini-3.5-flash-lite"}},
			},
		},
		products:   []string{"pro", "flash"},
		compounds:  map[string]string{"lite": "flash"},
		qualifiers: []string{"customtools"},
		stages:     []string{"preview"},
	},
	{
		prefix: "gemma-",
		lines: map[string][]generation{"": {
			{3, 0, map[string]string{"": "gemma-3-27b-it"}}, {4, 0, map[string]string{"": "gemma-4-26b-a4b-it"}},
		}},
		qualifiers: []string{"it"},
		sizes:      true,
	},
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
				{2, 5, map[string]string{"": "minimax-m2.5"}}, {2, 7, map[string]string{"": "minimax-m2.7"}},
				{3, 0, map[string]string{"": "minimax-m3"}},
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
			only(4, 30, "grok-4.3"), only(4, 50, "grok-4.5"), only(4, 60, "grok-4.6"), only(4, 70, "grok-4.7"),
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

func (f *lineFamily) parse(name string) (parsedName, bool) {
	name, _, _ = strings.Cut(name, ":")
	rest, ok := strings.CutPrefix(name, f.prefix)
	if !ok {
		return parsedName{}, false
	}
	tokens := strings.Split(rest, "-")
	version, hundredths := tokens[0], "00"
	if whole, fraction, dotted := strings.Cut(version, "."); f.decimalMinor && dotted {
		fraction = cmp.Or(strings.TrimRight(fraction, "0"), "0")
		version, hundredths = whole+"."+fraction, (fraction + "0")[:2]
	}
	major, minor, ok := parseVersion(version)
	if !ok {
		return parsedName{}, false
	}
	if f.decimalMinor {
		minor, _ = strconv.Atoi(hundredths)
	}
	p := parsedName{major: major, minor: minor}
	core := []string{f.prefix + version}
	inDate := false
	for i, token := range tokens[1:] {
		switch {
		case i == 0 && f.dashMinor && !strings.Contains(version, ".") && len(token) <= 2 && isDigits(token):
			p.minor, _ = strconv.Atoi(token)
			p.dashMinor = true
			core = append(core, token)
		case isDigits(token) && (len(token) >= 2 || inDate):
			inDate = true
		case token == "latest", slices.Contains(f.stages, token):
		case p.product == "" && slices.Contains(f.products, token):
			p.product = token
			core = append(core, token)
		case p.product != "" && f.compounds[token] == p.product:
			p.product += "-" + token
			core = append(core, token)
		case f.sizes && parameterCount.MatchString(token):
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

func (f *lineFamily) follow(name string) (p parsedName, line []generation, g generation, listed, ok bool) {
	if p, ok = f.parse(name); !ok {
		return p, nil, g, false, false
	}
	if line, ok = f.lines[p.product]; !ok {
		return p, nil, g, false, false
	}
	g, listed, ok = nearest(line, p.major, p.minor)
	return p, line, g, listed, ok
}

func (f *lineFamily) inherit(name string) (string, bool) {
	p, _, g, listed, ok := f.follow(name)
	if !ok || listed && p.qualifier != "" {
		return "", false
	}
	documented := p.spell(g)
	if release, _ := f.parse(documented); documented == name || release.core == p.core {
		return "", false
	}
	return documented, true
}

func (p parsedName) spell(g generation) string {
	id := g.member(p.qualifier)
	if !p.dashMinor {
		return id
	}
	return strings.Replace(id, fmt.Sprintf("%d.%d", g.major, g.minor), fmt.Sprintf("%d-%d", g.major, g.minor), 1)
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

var parameterCount = regexp.MustCompile(`^a?\d+(\.\d+)?[bt]$`)

func parseVersion(token string) (major, minor int, ok bool) {
	majorText, minorText, dotted := strings.Cut(token, ".")
	if !isDigits(majorText) || dotted && !isDigits(minorText) {
		return 0, 0, false
	}
	major, _ = strconv.Atoi(majorText)
	if dotted {
		minor, _ = strconv.Atoi(minorText)
	}
	return major, minor, true
}

func inheritLine(bare string) (string, bool) {
	for i := range lineFamilies {
		if documented, ok := lineFamilies[i].inherit(bare); ok {
			return documented, true
		}
	}
	return "", false
}

// InheritedOffWire is the disable a door sends for a model that inherits a
// refusal to switch thinking off: the wire of the newest release of its line that
// documents one, or OffOmit when none does.
func InheritedOffWire(model string, p Provider) OffWire {
	for _, release := range releasesBelow(model) {
		if off := ResolveOff(release, p); off != OffUnsupported {
			return off
		}
	}
	return OffOmit
}

func releasesBelow(model string) []string {
	documented, inherited := InheritedModel(model)
	if !inherited {
		return nil
	}
	_, _, bare := splitModelName(model)
	frame, framed := strings.CutSuffix(strings.ToLower(model), bare)
	if !framed {
		frame = model[:strings.LastIndex(model, "/")+1]
	}
	var below []string
	if tier, major, minor, ok := claudeVersion(documented); ok {
		line := claudeReleases[tier]
		g, _, _ := nearest(line, major, minor)
		for _, release := range atOrBelow(line, g) {
			below = append(below, frame+release.members[""])
		}
		return below
	}
	for i := range lineFamilies {
		p, line, g, _, ok := lineFamilies[i].follow(bare)
		if !ok {
			continue
		}
		for _, release := range atOrBelow(line, g) {
			if id := p.spell(release); id != "" {
				below = append(below, frame+id)
			}
		}
		break
	}
	return below
}
