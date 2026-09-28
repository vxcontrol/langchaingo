package openai_test

import "github.com/vxcontrol/langchaingo/llms"

func warningsOf(r *llms.ContentResponse) any { return r.Warnings }
