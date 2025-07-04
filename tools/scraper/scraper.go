package scraper

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/vxcontrol/langchaingo/tools"

	"github.com/gocolly/colly/v2"
)

const (
	DefualtMaxDept   = 1
	DefualtParallels = 3
	DefualtDelay     = 1000 * time.Millisecond
	DefualtAsync     = true
)

var ErrScrapingFailed = errors.New("scraper could not read URL, or scraping is not allowed for provided URL")

type Scraper struct {
	MaxDepth  int
	Parallels int
	Delay     time.Duration
	Blacklist []string
	Async     bool
}

var _ tools.Tool = Scraper{}

// New creates a new instance of Scraper with the provided options.
//
// The options parameter is a variadic argument allowing the user to specify
// custom configuration options for the Scraper. These options can be
// functions that modify the Scraper's properties.
//
// The function returns a pointer to a Scraper instance and an error. The
// error value is nil if the Scraper is created successfully.
func New(options ...Options) (*Scraper, error) {
	scraper := &Scraper{
		MaxDepth:  DefualtMaxDept,
		Parallels: DefualtParallels,
		Delay:     DefualtDelay,
		Async:     DefualtAsync,
		Blacklist: []string{
			"login",
			"signup",
			"signin",
			"register",
			"logout",
			"download",
			"redirect",
		},
	}

	for _, opt := range options {
		opt(scraper)
	}

	return scraper, nil
}

// Name returns the name of the Scraper.
//
// No parameters.
// Returns a string.
func (s Scraper) Name() string {
	return "Web Scraper"
}

// Description returns the description of the Go function.
//
// There are no parameters.
// It returns a string.
func (s Scraper) Description() string {
	return `
		Web Scraper will scan a url and return the content of the web page.
		Input should be a working url.
	`
}

// Call scrapes a website and returns the site data.
//
// The function takes a context.Context object for managing the execution
// context and a string input representing the URL of the website to be scraped.
// It returns a string containing the scraped data and an error if any.
//
//nolint:all
func (s Scraper) Call(ctx context.Context, input string) (string, error) {
	_, err := url.ParseRequestURI(input)
	if err != nil {
		return "", fmt.Errorf("%s: %w", ErrScrapingFailed, err)
	}

	c := colly.NewCollector(
		colly.MaxDepth(s.MaxDepth),
		colly.Async(s.Async),
	)

	err = c.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: s.Parallels,
		Delay:       s.Delay,
	})
	if err != nil {
		return "", fmt.Errorf("%s: %w", ErrScrapingFailed, err)
	}

	var siteData strings.Builder
	homePageLinks := make(map[string]bool)
	scrapedLinks := sync.Map{}

	writerMutex := sync.Mutex{}
	writeToBuffer := func(data string) {
		writerMutex.Lock()
		defer writerMutex.Unlock()

		siteData.WriteString(data)
	}

	c.OnRequest(func(r *colly.Request) {
		if ctx.Err() != nil {
			r.Abort()
		}
	})

	c.OnHTML("html", func(e *colly.HTMLElement) {
		currentURL := e.Request.URL.String()

		// Only process the page if it hasn't been visited yet
		if _, ok := scrapedLinks.LoadOrStore(currentURL, true); !ok {
			writeToBuffer("\n\nPage URL: " + currentURL)

			title := e.ChildText("title")
			if title != "" {
				writeToBuffer("\nPage Title: " + title)
			}

			description := e.ChildAttr("meta[name=description]", "content")
			if description != "" {
				writeToBuffer("\nPage Description: " + description)
			}

			writeToBuffer("\nHeaders:")
			e.ForEach("h1, h2, h3, h4, h5, h6", func(_ int, el *colly.HTMLElement) {
				writeToBuffer("\n" + el.Text)
			})

			writeToBuffer("\nContent:")
			e.ForEach("p", func(_ int, el *colly.HTMLElement) {
				writeToBuffer("\n" + el.Text)
			})

			if currentURL == input {
				e.ForEach("a", func(_ int, el *colly.HTMLElement) {
					link := el.Attr("href")
					if link != "" && !homePageLinks[link] {
						homePageLinks[link] = true
						writeToBuffer("\nLink: " + link)
					}
				})
			}
		}
	})

	c.OnHTML("a[href]", func(e *colly.HTMLElement) {
		link := e.Attr("href")
		absoluteLink := e.Request.AbsoluteURL(link)

		// Parse the link to get the hostname
		u, err := url.Parse(absoluteLink)
		if err != nil {
			// Handle the error appropriately
			return
		}

		// Check if the link's hostname matches the current request's hostname
		if u.Hostname() != e.Request.URL.Hostname() {
			return
		}

		// Check for redundant pages
		for _, item := range s.Blacklist {
			if strings.Contains(u.Path, item) {
				return
			}
		}

		// Normalize the path to treat '/' and '/index.html' as the same path
		if u.Path == "/index.html" || u.Path == "" {
			u.Path = "/"
		}

		// Only visit the page if it hasn't been visited yet
		if _, ok := scrapedLinks.LoadOrStore(u.String(), true); !ok {
			err := c.Visit(u.String())
			if err != nil {
				writeToBuffer(fmt.Sprintf("\nError following link %s: %v", link, err))
			}
		}
	})

	err = c.Visit(input)
	if err != nil {
		return "", fmt.Errorf("%s: %w", ErrScrapingFailed, err)
	}

	select {
	case <-ctx.Done():
		return "", ctx.Err()
	default:
		c.Wait()
	}

	// Append all scraped links
	writeToBuffer("\n\nScraped Links:")
	scrapedLinks.Range(func(key, value interface{}) bool {
		if link, ok := key.(string); ok {
			writeToBuffer("\n" + link)
		}
		return true
	})

	writerMutex.Lock()
	defer writerMutex.Unlock()

	return siteData.String(), nil
}
