package llms_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func everyCatalogueOption() []llms.CallOption {
	return []llms.CallOption{
		llms.WithMinP(0.05), llms.WithRepetitionPenalty(1.1),
		llms.WithFrequencyPenalty(0.3), llms.WithPresencePenalty(0.7),
		llms.WithTopK(40), llms.WithN(2), llms.WithCandidateCount(3),
		llms.WithTopLogProbs(5), llms.WithMinLength(10), llms.WithMaxLength(20),
		llms.WithSeed(7), llms.WithVerbosity("low"),
		llms.WithResponseMIMEType("application/json"),
		llms.WithLogProbs(true), llms.WithJSONMode(),
	}
}

var catalogueOptionNames = []string{
	"WithMinP", "WithRepetitionPenalty", "WithFrequencyPenalty", "WithPresencePenalty",
	"WithTopK", "WithN", "WithCandidateCount", "WithTopLogProbs",
	"WithMinLength", "WithMaxLength", "WithSeed", "WithVerbosity",
	"WithResponseMIMEType", "WithLogProbs", "WithJSONMode",
}

func TestTheUnreadCatalogueReportsEveryOptionADoorDoesNotCarry(t *testing.T) {
	t.Parallel()

	opts := llms.CallOptions{}
	for _, apply := range everyCatalogueOption() {
		apply(&opts)
	}

	var warn llms.Warnings
	warn.AddUnreadOptions("m", opts, "no field", "WithSeed", "WithTopK")

	reported := make(map[string]llms.Warning, len(catalogueOptionNames))
	for _, w := range warn.List() {
		reported[w.Option] = w
	}
	for _, option := range catalogueOptionNames {
		switch option {
		case "WithSeed", "WithTopK":
			require.NotContains(t, reported, option, "the door carries it")
		default:
			w, ok := reported[option]
			require.True(t, ok, "%s went unreported", option)
			require.Equal(t, llms.WarningDrop, w.Kind)
			require.NotEmpty(t, w.Asked, "%s reported without the value asked", option)
		}
	}
}

func TestADoorThatCarriesNothingReportsTheWholeCatalogue(t *testing.T) {
	t.Parallel()

	opts := llms.CallOptions{}
	for _, apply := range everyCatalogueOption() {
		apply(&opts)
	}
	for _, apply := range []llms.CallOption{
		llms.WithTemperature(0.4), llms.WithTopP(0.9), llms.WithMaxTokens(1024),
		llms.WithStopWords([]string{"stop"}),
	} {
		apply(&opts)
	}

	var warn llms.Warnings
	warn.AddUnreadOptions("m", opts, "no field")

	reported := make([]string, 0, len(catalogueOptionNames))
	for _, w := range warn.List() {
		reported = append(reported, w.Option)
	}
	require.ElementsMatch(t, catalogueOptionNames, reported,
		"the catalogue holds the options a door carries verbatim or not at all; "+
			"an option the door reshapes is reported with the value that travelled")
}

func TestACallThatSetsNoCatalogueOptionReportsNothing(t *testing.T) {
	t.Parallel()

	var warn llms.Warnings
	warn.AddUnreadOptions("m", llms.CallOptions{}, "no field")

	require.Empty(t, warn.List())
}
