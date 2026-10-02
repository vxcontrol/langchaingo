package bedrock_test

import (
	"context"
	"io"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
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

type bodyKindRecorder struct {
	mu       sync.Mutex
	writerTo map[string]bool
}

func (r *bodyKindRecorder) Do(req *http.Request) (*http.Response, error) {
	_, writerTo := req.Body.(io.WriterTo)
	r.mu.Lock()
	r.writerTo[req.URL.EscapedPath()] = writerTo
	r.mu.Unlock()
	return &http.Response{
		StatusCode: http.StatusBadRequest,
		Header:     http.Header{"X-Amzn-Errortype": {"ValidationException"}, "Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(`{"message":"recorded"}`)),
		Request:    req,
	}, nil
}

func TestEveryCallOfTheDoorHandsTheTransportABodyWithoutWriteTo(t *testing.T) {
	t.Parallel()

	recorder := &bodyKindRecorder{writerTo: map[string]bool{}}
	client := bedrockruntime.NewFromConfig(aws.Config{
		Region:      "us-east-1",
		Credentials: credentials.NewStaticCredentialsProvider("unit", "test", ""),
		HTTPClient:  recorder,
	}, func(o *bedrockruntime.Options) { o.BaseEndpoint = aws.String("https://bedrock.test") }, signWithSigV4)

	ask := func(model string, converse bool) {
		opts := []bedrock.Option{bedrock.WithClient(client), bedrock.WithModel(model)}
		if converse {
			opts = append(opts, bedrock.WithConverseAPI())
		}
		llm, err := bedrock.New(opts...)
		require.NoError(t, err)
		for _, callOpts := range [][]llms.CallOption{
			nil, {llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil })},
		} {
			_, err := llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, callOpts...)
			require.ErrorContains(t, err, "recorded", model)
		}
	}
	legacy := []string{
		"ai21.j2-ultra-v1", "ai21.jamba-1-5-large-v1:0", "amazon.titan-text-express-v1", "amazon.nova-pro-v1:0",
		"anthropic.claude-sonnet-4-5-20250929-v1:0", "cohere.command-text-v14", "cohere.command-r-v1:0",
		"meta.llama3-70b-instruct-v1:0", "deepseek.r1-v1:0",
	}
	for _, model := range legacy {
		ask(model, false)
	}
	ask("anthropic.claude-sonnet-4-5-20250929-v1:0", true)

	want := make([]string, 0, 2+2*len(legacy))
	want = append(want, "/model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse",
		"/model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse-stream")
	for _, model := range legacy {
		escaped := strings.ReplaceAll(model, ":", "%3A")
		want = append(want, "/model/"+escaped+"/invoke", "/model/"+escaped+"/invoke-with-response-stream")
	}
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	require.ElementsMatch(t, want, slices.Collect(maps.Keys(recorder.writerTo)))
	for call, writerTo := range recorder.writerTo {
		require.False(t, writerTo, call)
	}
}
