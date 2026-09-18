package reasoning

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"maps"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

const reasoningImportPath = "github.com/vxcontrol/langchaingo/llms/reasoning"

func providerConstantsPassedIn(t *testing.T, dir string) map[string]bool {
	t.Helper()

	passed := map[string]bool{}
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		switch {
		case err != nil:
			return err
		case d.IsDir() && d.Name() == "testdata":
			return filepath.SkipDir
		case d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go"):
			return nil
		}
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		local := ""
		for _, spec := range file.Imports {
			if importPath, _ := strconv.Unquote(spec.Path.Value); importPath == reasoningImportPath {
				local = "reasoning"
				if spec.Name != nil {
					local = spec.Name.Name
				}
			}
		}
		if local == "" {
			return nil
		}
		ast.Inspect(file, func(n ast.Node) bool {
			sel, ok := n.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == local &&
				strings.HasPrefix(sel.Sel.Name, "Provider") && sel.Sel.Name != "Provider" {
				passed[sel.Sel.Name] = true
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatalf("reading %s: %v", dir, err)
	}
	return passed
}

func TestEveryProviderConstantHasADoorThatPassesIt(t *testing.T) {
	t.Parallel()

	doors := map[string]string{
		"ProviderAnthropic": "../anthropic",
		"ProviderBedrock":   "../bedrock",
		"ProviderOpenAI":    "../openai",
		"ProviderGoogleAI":  "../googleai",
		"ProviderOllama":    "../ollama",
	}

	if len(doors) != int(providerCount)-1 {
		t.Fatalf("the enum holds %d door providers besides ProviderUnknown, this test pairs %d of them with a door",
			int(providerCount)-1, len(doors))
	}
	for constant, dir := range doors {
		passed := providerConstantsPassedIn(t, dir)
		if len(passed) != 1 || !passed[constant] {
			t.Errorf("%s passes %v; it must pass %s and no other door's constant",
				dir, slices.Sorted(maps.Keys(passed)), constant)
		}
	}
}

func TestOllamaDisablesByBooleanEvenForNamesOtherVendorsServeAsMandatory(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"deepseek-r1:7b", "qwen3:8b", "magistral-small:24b"} {
		if got := ResolveOff(model, ProviderOllama); got != OffDisableThinkBool {
			t.Errorf("ResolveOff(%q, ProviderOllama) = %v, want OffDisableThinkBool", model, got)
		}
		if same := ResolveOff(model, ProviderUnknown); same == OffDisableThinkBool {
			t.Errorf("ResolveOff(%q, ProviderUnknown) already answers for the ollama door", model)
		}
	}
}

func TestOllamaRefusesToDisableTheFamilyThatIgnoresBooleans(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-oss:120b", "gpt-oss:20b", "library/gpt-oss:120b"} {
		if got := ResolveOff(model, ProviderOllama); got != OffUnsupported {
			t.Errorf("ResolveOff(%q, ProviderOllama) = %v, want OffUnsupported", model, got)
		}
	}
}

func TestEveryMistralModelDisablesByOmission(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"mistral-small-latest", "mistral-medium-latest",
		"mistral-large-latest", "magistral-medium-latest",
	} {
		if got := ResolveOff(model, ProviderUnknown); got != OffOmit {
			t.Errorf("ResolveOff(%q) = %v, want OffOmit", model, got)
		}
	}
}

func TestOnlyOpenAIsOwnCatalogueRefusesTopK(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-4o", "gpt-5.4-nano", "o3", "o4-mini", "chatgpt-4o-latest"} {
		if !RejectsTopK(model) {
			t.Errorf("%q is OpenAI's own name; its endpoint fails the call on top_k", model)
		}
	}
	for _, model := range []string{"gpt-oss:20b", "gpt-oss-120b", "zai/glm-4.5-air", "qwen3-max", "grok-4"} {
		if RejectsTopK(model) {
			t.Errorf("%q is not an OpenAI name and must not inherit its refusal", model)
		}
	}
}
