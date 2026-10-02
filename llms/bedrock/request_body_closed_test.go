package bedrock_test

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	awshttp "github.com/aws/aws-sdk-go-v2/aws/transport/http"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type bodyHoldingConn struct {
	net.Conn
	writes    int
	resume    <-chan struct{}
	closed    chan struct{}
	closeOnce sync.Once
}

func (c *bodyHoldingConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	if c.writes++; c.writes == 2 {
		<-c.resume
	}
	return n, err
}

func (c *bodyHoldingConn) Close() error {
	c.closeOnce.Do(func() { close(c.closed) })
	return c.Conn.Close()
}

func streamWithTheRequestBodyHeld(
	t *testing.T, firstHalf, secondHalf func(http.ResponseWriter, *eventstream.Encoder), opts ...bedrock.Option,
) (*llms.ContentResponse, error) {
	t.Helper()

	resume, rest := make(chan struct{}), make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		enc := eventstream.NewEncoder()
		firstHalf(w, enc)
		w.(http.Flusher).Flush()
		<-rest
		secondHalf(w, enc)
	}))
	t.Cleanup(srv.Close)

	conns := make(chan *bodyHoldingConn, 1)
	httpClient := awshttp.NewBuildableClient().WithTransportOptions(func(tr *http.Transport) {
		dial := tr.DialContext
		tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			c, err := dial(ctx, network, addr)
			if err != nil {
				return nil, err
			}
			held := &bodyHoldingConn{Conn: c, resume: resume, closed: make(chan struct{})}
			select {
			case conns <- held:
			default:
			}
			return held, nil
		}
	})
	client := bedrockruntime.NewFromConfig(aws.Config{
		Region:      "us-east-1",
		Credentials: credentials.NewStaticCredentialsProvider("unit", "test", ""),
		HTTPClient:  httpClient,
	}, func(o *bedrockruntime.Options) { o.BaseEndpoint = aws.String(srv.URL) }, signWithSigV4)
	llm, err := bedrock.New(append([]bedrock.Option{bedrock.WithClient(client)}, opts...)...)
	require.NoError(t, err)

	var firstChunk sync.Once
	return llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error {
			firstChunk.Do(func() {
				close(resume)
				select {
				case <-(<-conns).closed:
				case <-time.After(200 * time.Millisecond):
				}
				close(rest)
			})
			return nil
		}))
}

func TestALegacyStreamOutlivesTheSDKClosingItsRequestBody(t *testing.T) {
	t.Parallel()

	resp, err := streamWithTheRequestBodyHeld(t,
		func(w http.ResponseWriter, enc *eventstream.Encoder) {
			writeLegacyChunk(t, w, enc, `{"type":"message_start","message":{"id":"x","type":"message",`+
				`"role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}`)
			writeLegacyChunk(t, w, enc, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"first "}}`)
		},
		func(w http.ResponseWriter, enc *eventstream.Encoder) {
			writeLegacyChunk(t, w, enc, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"second"}}`)
			writeLegacyChunk(t, w, enc, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`)
		},
		bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"))
	require.NoError(t, err)
	require.Equal(t, "first second", resp.Choices[0].Content)
}

func TestAConverseStreamOutlivesTheSDKClosingItsRequestBody(t *testing.T) {
	t.Parallel()

	resp, err := streamWithTheRequestBodyHeld(t,
		func(w http.ResponseWriter, enc *eventstream.Encoder) {
			writeConverseEvent(t, w, enc, "messageStart", `{"role":"assistant"}`)
			writeConverseEvent(t, w, enc, "contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"first "}}`)
		},
		func(w http.ResponseWriter, enc *eventstream.Encoder) {
			writeConverseEvent(t, w, enc, "contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"second"}}`)
			writeConverseEvent(t, w, enc, "contentBlockStop", `{"contentBlockIndex":0}`)
			writeConverseEvent(t, w, enc, "messageStop", `{"stopReason":"end_turn"}`)
			writeConverseEvent(t, w, enc, "metadata", `{"usage":{"inputTokens":1,"outputTokens":2,"totalTokens":3}}`)
		},
		bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
	require.NoError(t, err)
	require.Equal(t, "first second", resp.Choices[0].Content)
}
