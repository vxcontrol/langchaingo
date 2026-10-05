package llms

import (
	"bufio"
	"cmp"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

var updateCatalogueSnapshot = flag.Bool("update-catalogue-snapshot", false,
	"rewrite testdata/pentagi_catalogues.tsv from what the tables answer now")

var catalogueProviders = map[string]reasoning.Provider{
	"anthropic": reasoning.ProviderAnthropic,
	"bedrock":   reasoning.ProviderBedrock,
	"gemini":    reasoning.ProviderGoogleAI,
}

// catalogueHosts mirrors the default server URLs in PentAGI's config for the
// catalogues whose host changes what the tables answer.
var catalogueHosts = map[string]string{"qwen": "dashscope-us.aliyuncs.com"}

var offWireNames = []string{
	"omit", "disable-claude", "zero-budget", "minimal-level", "effort-none", "disable-dashscope",
	"disable-thinking-object", "disable-think-bool", "between-tools-claude", "unsupported",
}

func catalogueRow(catalogue, model string) string {
	provider, ok := catalogueProviders[catalogue]
	if !ok {
		provider = reasoning.ProviderOpenAI
	}
	support := ReasoningSupportFor(model, provider)
	efforts := make([]string, 0, len(support.Efforts))
	for _, effort := range support.Efforts {
		efforts = append(efforts, string(effort))
	}
	defaultOn := "-"
	if support.DefaultOn != nil {
		defaultOn = fmt.Sprint(*support.DefaultOn)
	}
	off := reasoning.ResolveOff(model, provider)
	return strings.Join([]string{
		catalogue, model,
		fmt.Sprint(support.Supported), fmt.Sprint(support.Known), fmt.Sprint(support.CannotDisable),
		fmt.Sprint(support.RejectsSampling), fmt.Sprint(int(support.Mechanism)),
		strings.Join(efforts, ","), defaultOn, offWireNames[off],
		fmt.Sprint(int(reasoning.EffortWithTools(model))),
		fmt.Sprint(provider == reasoning.ProviderOpenAI &&
			reasoning.TakesNoThinkingDepth(reasoning.DashScopeRoute(model, catalogueHosts[catalogue]))),
		cmp.Or(support.Inherited, "-"),
	}, "\t")
}

func TestPentagiCataloguesReadTheTablesAsRecorded(t *testing.T) {
	path := filepath.Join("testdata", "pentagi_catalogues.tsv")
	file, err := os.Open(path)
	require.NoError(t, err)
	defer file.Close()

	var header string
	var recorded, now []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if header == "" {
			header = line
			continue
		}
		recorded = append(recorded, line)
		catalogue, rest, _ := strings.Cut(line, "\t")
		model, _, _ := strings.Cut(rest, "\t")
		now = append(now, catalogueRow(catalogue, model))
	}
	require.NoError(t, scanner.Err())

	if *updateCatalogueSnapshot {
		require.NoError(t, os.WriteFile(path, []byte(header+"\n"+strings.Join(now, "\n")+"\n"), 0o600))
		return
	}
	for i := range recorded {
		require.Equal(t, recorded[i], now[i])
	}
}
