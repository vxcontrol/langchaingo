package main

import (
	"fmt"
	"net/url"
)

func main() {
	u, err := url.Parse("us-central1-aiplatform.googleapis.com:443")
	fmt.Println(err)
	if err == nil {
		fmt.Println(u.JoinPath("v1beta1", "projects/p/x").String())
	}
}
