package googleai

import (
	"context"
	"strconv"

	"google.golang.org/genai"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type geminiReport struct {
	reported       bool
	thinking       bool
	topK           int32
	maxTemperature float32
}

func (g *GoogleAI) geminiReport(ctx context.Context, model string) geminiReport {
	if g.client.ClientConfig().Backend != genai.BackendGeminiAPI || reasoning.GeminiDocumented(model) {
		return geminiReport{}
	}
	return g.models.Get(ctx, model, func(ctx context.Context) (geminiReport, error) {
		m, err := g.client.Models.Get(ctx, model, nil)
		if err != nil {
			return geminiReport{}, err
		}
		return geminiReportFrom(m), nil
	})
}

func geminiReportFrom(m *genai.Model) geminiReport {
	if m.Name == "" || m.InputTokenLimit == 0 {
		return geminiReport{}
	}
	return geminiReport{reported: true, thinking: m.Thinking, topK: m.TopK, maxTemperature: m.MaxTemperature}
}

type geminiRules struct {
	model  string
	lookup func() geminiReport
	asked  *geminiReport
}

func (r *geminiRules) report() geminiReport {
	if r.asked == nil {
		report := r.lookup()
		r.asked = &report
	}
	return *r.asked
}

func (r *geminiRules) addInherited(warn *llms.Warnings) {
	if r.asked != nil && r.asked.reported {
		warn.AddInheritedReported(r.model)
		return
	}
	warn.AddInherited(r.model)
}

func (r *geminiRules) thinksNever() bool {
	return r.report().reported && !r.report().thinking
}

func (r *geminiRules) thinks() bool {
	return r.report().reported && r.report().thinking
}

func (r *geminiRules) fitSampling(warn *llms.Warnings, opts llms.CallOptions, config *genai.GenerateContentConfig) {
	if opts.TopK != nil && r.report().reported && r.report().topK == 0 {
		config.TopK = nil
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithTopK", Model: r.model, Asked: strconv.Itoa(*opts.TopK),
			Reason: "the API reports that the model takes no top-k sampling",
		})
	}
	if opts.Temperature == nil || !r.report().reported {
		return
	}
	if limit := r.report().maxTemperature; limit > 0 && *config.Temperature > limit {
		config.Temperature = &limit
		warn.Add(llms.Warning{
			Kind: llms.WarningClamp, Option: "WithTemperature", Model: r.model,
			Asked: strconv.FormatFloat(*opts.Temperature, 'g', -1, 64), Sent: strconv.FormatFloat(float64(limit), 'g', -1, 32),
			Reason: "the API reports a lower maximum temperature for the model",
		})
	}
}
