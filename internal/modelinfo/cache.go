// Package modelinfo keeps what a vendor's API reports about a model for a while,
// so a door asks the vendor at most once per model and TTL.
package modelinfo

import (
	"context"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
)

const DefaultTTL = time.Hour

const lookupTimeout = 5 * time.Second

type entry[T any] struct {
	value T
	until time.Time
}

type Cache[T any] struct {
	ttl     time.Duration
	now     func() time.Time
	group   singleflight.Group
	mu      sync.Mutex
	entries map[string]entry[T]
}

func New[T any](ttl time.Duration) *Cache[T] {
	if ttl <= 0 {
		ttl = DefaultTTL
	}
	return &Cache[T]{ttl: ttl, now: time.Now, entries: map[string]entry[T]{}}
}

func (c *Cache[T]) Get(ctx context.Context, key string, lookup func(context.Context) (T, error)) T {
	var zero T
	if c == nil {
		return zero
	}
	if value, ok := c.fresh(key); ok {
		return value
	}
	done := c.group.DoChan(key, func() (any, error) {
		if value, ok := c.fresh(key); ok {
			return value, nil
		}
		lookupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), lookupTimeout)
		defer cancel()
		value, err := lookup(lookupCtx)
		c.mu.Lock()
		defer c.mu.Unlock()
		if err != nil {
			value = c.entries[key].value
		}
		c.entries[key] = entry[T]{value: value, until: c.now().Add(c.ttl)}
		return value, nil
	})
	wait := lookupTimeout
	if deadline, ok := ctx.Deadline(); ok {
		wait = min(wait, time.Until(deadline)/2)
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return zero
	case <-timer.C:
		return zero
	case result := <-done:
		value, _ := result.Val.(T)
		return value
	}
}

func (c *Cache[T]) fresh(key string) (T, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key]
	if !ok || !c.now().Before(e.until) {
		var zero T
		return zero, false
	}
	return e.value, true
}
