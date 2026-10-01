package cutctx

import (
	"context"
	"sync"
)

type Context struct {
	context.Context
	err  error
	done chan struct{}
	once sync.Once
}

func New(parent context.Context, err error) *Context {
	return &Context{Context: parent, err: err, done: make(chan struct{})}
}

func (c *Context) Cut() {
	c.once.Do(func() { close(c.done) })
}

func (c *Context) Done() <-chan struct{} {
	return c.done
}

func (c *Context) Err() error {
	select {
	case <-c.done:
		return c.err
	default:
		return nil
	}
}
