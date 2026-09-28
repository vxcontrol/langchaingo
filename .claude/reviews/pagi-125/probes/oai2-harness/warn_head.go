package openai_test

import (
	"fmt"

	"github.com/vxcontrol/langchaingo/llms"
)

func printWarn(name string, resp *llms.ContentResponse) {
	if resp == nil {
		return
	}
	for _, w := range resp.Warnings {
		fmt.Printf("[%s] WARN: %s\n", name, w.String())
	}
}
