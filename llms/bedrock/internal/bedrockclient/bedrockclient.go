package bedrockclient

import (
	"context"
	"errors"
	"strconv"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// Client is a Bedrock client.
type Client struct {
	client legacyRuntime
}

// Message is a chunk of text or an data
// that will be sent to the provider.
//
// The provider may then transform the message to its own
// format before sending it to the LLM model API.
type Message struct {
	Role llms.ChatMessageType
	// Content contains the main message content
	Content string
	// Type may be "text", "image", "tool_use", "tool_result"
	Type string
	// MimeType is the MIME type for image content
	MimeType string
	// Tool calling fields
	ToolCall   *ToolCall   `json:"tool_call,omitempty"`
	ToolResult *ToolResult `json:"tool_result,omitempty"`
	// Reasoning contains thinking content for AI messages
	Reasoning *reasoning.ContentReasoning `json:"reasoning,omitempty"`
	// CacheControl for prompt caching (used by Legacy Anthropic API)
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// CacheControl represents cache configuration for prompt caching
type CacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// ToolCall represents a function call request from the model
type ToolCall struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// ToolResult represents the result of a function call execution
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	ToolName   string `json:"tool_name"`
	Content    string `json:"content"`
}

func GetProvider(modelID string) string {
	// Check for Nova models (including inference profiles like us.amazon.nova-*)
	if strings.Contains(modelID, ".nova-") || strings.Contains(modelID, "amazon.nova-") {
		return "nova"
	}

	parts := strings.Split(modelID, ".")

	// For backward compatibility with the original provider detection
	switch {
	case strings.Contains(modelID, "ai21"):
		return "ai21"
	case strings.Contains(modelID, "amazon"):
		return "amazon"
	case strings.Contains(modelID, "anthropic"):
		return "anthropic"
	case strings.Contains(modelID, "cohere"):
		return "cohere"
	case strings.Contains(modelID, "meta"):
		return "meta"
	case strings.Contains(modelID, "deepseek"):
		return "deepseek"
	}

	// Default to using the first part of the model ID
	if len(parts) > 0 {
		return parts[0]
	}

	return ""
}

// NewClient creates a new Bedrock client.
func NewClient(client *bedrockruntime.Client) *Client {
	return &Client{
		client: legacyReadOnlyBodies{client},
	}
}

// CreateCompletion creates a new completion response from the provider
// after sending the messages to the provider.
func (c *Client) CreateCompletion(ctx context.Context,
	modelID string,
	messages []Message,
	options llms.CallOptions,
) (*llms.ContentResponse, error) {
	provider := GetProvider(modelID)
	// Legacy InvokeModel structured output is implemented only for the Anthropic
	// payload; other providers get a typed error rather than a guessed wire shape.
	if options.StructuredOutput != nil && provider != "anthropic" {
		return nil, &llms.ErrStructuredOutputUnsupported{
			Provider: providerBedrock,
			Model:    modelID,
			Reason:   "legacy InvokeModel structured output is only implemented for Anthropic models; use the Converse API for other providers",
		}
	}
	warn := &llms.Warnings{}
	warn.AddInherited(modelID)
	if options.Reasoning.ResolveMode() == llms.ReasoningOff &&
		reasoning.ResolveOff(modelID, reasoning.ProviderBedrock) == reasoning.OffUnsupported &&
		warn.KeepOffRefusal(modelID, reasoning.InheritedOffWire(modelID, reasoning.ProviderBedrock)) {
		return nil, &reasoning.ErrReasoningOffUnsupported{Model: modelID}
	}
	reportLegacyOptions(warn, provider, modelID, options)

	var (
		resp *llms.ContentResponse
		err  error
	)
	switch provider {
	case "ai21":
		resp, err = createAi21Completion(ctx, c.client, modelID, messages, options, warn)
	case "amazon":
		resp, err = createAmazonCompletion(ctx, c.client, modelID, messages, options, warn)
	case "nova":
		resp, err = createNovaCompletion(ctx, c.client, modelID, messages, options, warn)
	case "anthropic":
		resp, err = createAnthropicCompletion(ctx, c.client, modelID, messages, options, warn)
	case "cohere":
		resp, err = createCohereCompletion(ctx, c.client, modelID, messages, options, warn)
	case "meta":
		resp, err = createMetaCompletion(ctx, c.client, modelID, messages, options, warn)
	case "deepseek":
		resp, err = createDeepSeekCompletion(ctx, c.client, modelID, messages, options, warn)
	default:
		return nil, errors.New("unsupported provider")
	}
	if resp != nil {
		resp.Warnings = warn.List()
	}
	return resp, err
}

// Helper function to process input text chat
// messages as a single string.
func processInputMessagesGeneric(messages []Message) string {
	var sb strings.Builder
	var hasRole bool
	for _, message := range messages {
		if message.Role != "" {
			hasRole = true
			sb.WriteString("\n")
			sb.WriteString(string(message.Role))
			sb.WriteString(": ")
		}
		if message.Type == "text" {
			sb.WriteString(message.Content)
		}
	}
	if hasRole {
		sb.WriteString("\n")
		sb.WriteString("AI: ")
	}
	return sb.String()
}

func IsAi21Jamba(modelID string) bool {
	return strings.Contains(modelID, "jamba")
}

func IsCohereCommandR(modelID string) bool {
	return strings.Contains(modelID, "command-r")
}

func maxTokensOnTheWire(
	warn *llms.Warnings, modelID string, options llms.CallOptions, defaultValue int,
) int {
	sent := answerLimit(modelID, options, defaultValue)
	reportAnswerLimit(warn, modelID, options, sent)
	return sent
}

func answerLimit(modelID string, options llms.CallOptions, defaultValue int) int {
	sent := getMaxTokens(options.GetMaxTokens(), defaultValue)
	if ceiling := answerCeiling(modelID); ceiling != 0 && sent > ceiling {
		return ceiling
	}
	return sent
}

func reportAnswerLimit(warn *llms.Warnings, modelID string, options llms.CallOptions, sent int) {
	asked := options.MaxTokens
	switch {
	case asked == nil || *asked == sent:
	case sent == 0:
		warn.Add(llms.Warning{
			Kind: llms.WarningDrop, Option: "WithMaxTokens", Model: modelID,
			Asked: strconv.Itoa(*asked), Reason: "the request names no answer limit, so the vendor's default applies",
		})
	case *asked <= 0:
		warn.Add(llms.Warning{
			Kind: llms.WarningSubstitute, Option: "WithMaxTokens", Model: modelID,
			Asked: strconv.Itoa(*asked), Sent: strconv.Itoa(sent),
			Reason: "a non-positive limit is replaced by the door's default for this payload",
		})
	default:
		warn.Add(llms.Warning{
			Kind: llms.WarningClamp, Option: "WithMaxTokens", Model: modelID,
			Asked: strconv.Itoa(*asked), Sent: strconv.Itoa(sent),
			Reason: "the model's documented answer limit is lower",
		})
	}
}

var answerCeilings = []struct {
	family  string
	ceiling int
}{
	{"amazon.titan-text-lite", 4096},
	{"amazon.titan-text-express", 8192},
	{"amazon.titan-text-premier", 3072},
	{"amazon.nova-micro", 5000},
	{"amazon.nova-lite", 5000},
	{"amazon.nova-pro", 5000},
	{"amazon.nova-premier", 5000},
	{"amazon.nova-2-lite", 64000},
	{"cohere.command-text", 4096},
	{"cohere.command-light-text", 4096},
	{"ai21.jamba", 4096},
	{"ai21.j2-mid", 8191},
	{"ai21.j2-ultra", 8191},
	{"ai21.j2", 2048},
	{"meta.llama", 2048},
	{"deepseek.r1", 32768},
}

func answerCeiling(modelID string) int {
	id := strings.ToLower(modelID)
	for _, entry := range answerCeilings {
		if strings.Contains(id, entry.family) {
			return entry.ceiling
		}
	}
	return 0
}

func getMaxTokens(maxTokens, defaultValue int) int {
	if maxTokens <= 0 {
		return defaultValue
	}
	return maxTokens
}
