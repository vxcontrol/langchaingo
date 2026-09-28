package googleai

import (
	"fmt"

	"cloud.google.com/go/auth"
	"cloud.google.com/go/auth/credentials"
)

const cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

func (o *Options) detectCredentials() (*auth.Credentials, error) {
	if o.credentialsFile == "" && len(o.credentialsJSON) == 0 {
		return nil, nil //nolint:nilnil // no credentials named is not an error
	}

	detected, err := credentials.DetectDefault(&credentials.DetectOptions{
		CredentialsFile: o.credentialsFile,
		CredentialsJSON: o.credentialsJSON,
		Scopes:          []string{cloudPlatformScope},
	})
	if err != nil {
		return nil, fmt.Errorf("googleai: the credentials the caller named cannot be read: %w", err)
	}

	return detected, nil
}
