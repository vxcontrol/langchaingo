package openai_test

import (
	"fmt"
	"github.com/vxcontrol/langchaingo/llms"
)

func printWarnVF(name string, resp *llms.ContentResponse) {
	if resp == nil {
		return
	}
	fmt.Printf("[%s] warnings=%d\n", name, len(resp.Warnings))
	for _, w := range resp.Warnings {
		fmt.Printf("[%s] WARN: %s\n", name, w.String())
	}
}
