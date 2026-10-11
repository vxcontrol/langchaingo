package openai

import (
	"context"
	"net/url"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type vendorReport struct {
	thinks          *bool
	efforts         []string
	effortsReported bool
}

func (r vendorReport) reported() bool {
	return r.thinks != nil || r.effortsReported
}

func (o *LLM) vendorReport(ctx context.Context, model string) vendorReport {
	vendor := reasoning.HostVendor(o.host)
	if o.models == nil || reasoning.DocumentedOn(model, vendor) {
		return vendorReport{}
	}
	var lookup func(context.Context, string) (vendorReport, error)
	switch vendor { //nolint:exhaustive // DocumentedOn answers true for every other vendor
	case reasoning.VendorXAI:
		lookup = o.xaiReport
	case reasoning.VendorMoonshot:
		lookup = o.moonshotReport
	case reasoning.VendorMistral:
		lookup = o.mistralReport
	default:
		return vendorReport{}
	}
	return o.models.Get(ctx, model, func(ctx context.Context) (vendorReport, error) {
		return lookup(ctx, model)
	})
}

func (o *LLM) xaiReport(ctx context.Context, model string) (vendorReport, error) {
	var m struct {
		Capabilities *struct {
			ReasoningEffort []string `json:"reasoning_effort"`
		} `json:"capabilities"`
	}
	if err := o.client.GetJSON(ctx, "/models/"+url.PathEscape(model), &m); err != nil || m.Capabilities == nil {
		return vendorReport{}, err
	}
	return vendorReport{efforts: m.Capabilities.ReasoningEffort, effortsReported: true}, nil
}

func (o *LLM) moonshotReport(ctx context.Context, model string) (vendorReport, error) {
	var list struct {
		Data []struct {
			ID                string `json:"id"`
			SupportsReasoning *bool  `json:"supports_reasoning"`
		} `json:"data"`
	}
	if err := o.client.GetJSON(ctx, "/models", &list); err != nil {
		return vendorReport{}, err
	}
	for _, m := range list.Data {
		if m.ID == model {
			return vendorReport{thinks: m.SupportsReasoning}, nil
		}
	}
	return vendorReport{}, nil
}

func (o *LLM) mistralReport(ctx context.Context, model string) (vendorReport, error) {
	var m struct {
		Capabilities *struct {
			Reasoning bool `json:"reasoning"`
		} `json:"capabilities"`
	}
	if err := o.client.GetJSON(ctx, "/models/"+url.PathEscape(model), &m); err != nil || m.Capabilities == nil {
		return vendorReport{}, err
	}
	return vendorReport{thinks: &m.Capabilities.Reasoning}, nil
}

type vendorRules struct {
	model  string
	vendor reasoning.Vendor
	lookup func() vendorReport
	asked  *vendorReport
}

func (r *vendorRules) report() vendorReport {
	if r == nil {
		return vendorReport{}
	}
	if r.asked == nil {
		var report vendorReport
		if r.lookup != nil {
			report = r.lookup()
		}
		r.asked = &report
	}
	return *r.asked
}

func (r *vendorRules) addInherited(warn *llms.Warnings) {
	if r.asked != nil && r.asked.reported() {
		warn.AddInheritedReported(r.model)
		return
	}
	warn.AddInherited(r.model)
}

func (r *vendorRules) thinks() (thinks, reported bool) {
	if r == nil || r.vendor != reasoning.VendorMoonshot && r.vendor != reasoning.VendorMistral {
		return false, false
	}
	if t := r.report().thinks; t != nil {
		return *t, true
	}
	return false, false
}

func (r *vendorRules) thinksNever() bool {
	thinks, reported := r.thinks()
	return reported && !thinks
}

func (r *vendorRules) mistralReasons() bool {
	if r == nil || !reasoning.ServedByMistral(r.model) || reasoning.IsReasoningModel(r.model) {
		return false
	}
	thinks, _ := r.thinks()
	return thinks
}

func (r *vendorRules) efforts() ([]string, bool) {
	report := r.report()
	return report.efforts, report.effortsReported
}
