> ⚠️ **This is a fork** of the original [github.com/tmc/langchaingo](https://github.com/tmc/langchaingo) repository.

# 🦜️🔗 LangChain Go (fork)

[![go.dev reference](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white&style=flat-square)](https://pkg.go.dev/github.com/vxcontrol/langchaingo)
[![scorecard](https://goreportcard.com/badge/github.com/vxcontrol/langchaingo)](https://goreportcard.com/report/github.com/vxcontrol/langchaingo)
[![](https://dcbadge.vercel.app/api/server/8bHGKzHBkM?compact=true&style=flat)](https://discord.gg/8bHGKzHBkM)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/vxcontrol/langchaingo)
[<img src="https://github.com/codespaces/badge.svg" title="Open in Github Codespace" width="150" height="20">](https://codespaces.new/vxcontrol/langchaingo)

⚡ Building applications with LLMs through composability, with Go! ⚡

## Why this fork?

This fork was created to incorporate functionality from open Pull Requests that haven't been merged into the original repository yet. You can view the list of accepted PRs in our [v0.1.13-update.0 release](https://github.com/vxcontrol/langchaingo/releases/tag/v0.1.13-update.0) and in the [main-pull-requests branch](https://github.com/vxcontrol/langchaingo/commits/main-pull-requests/).

Additionally, this repository contains custom improvements and enhancements related to langchaingo support, which will be published as releases. The original repository's state can be accessed in the [main branch](https://github.com/vxcontrol/langchaingo/tree/main), which will be regularly updated with upstream changes.

This fork is primarily maintained for use in the [PentAGI](https://github.com/vxcontrol/pentagi) project, an autonomous AI Agents system for performing complex penetration testing tasks.

## Branch Structure and Versioning

This repository follows a specific branching strategy to maintain both upstream compatibility and custom enhancements:

### Branch Management

- **main**: Fully synchronized with upstream (`tmc/langchaingo`). Never force-pushed.
- **main-pull-requests**: Contains merged PRs from upstream that haven't been officially merged. Rebased on `main` after synchronization (commit hashes will change).
- **main-vxcontrol**: Default branch containing all current enhancements. Rebased on `main-pull-requests` (commit hashes will change).
- **release/v***: Created from `main-vxcontrol` for each release. These branches are stable and never force-pushed.

### Versioning

Release tags follow the format `v0.1.13-update.1`, where:
- `v0.1.13` corresponds to the latest upstream release version
- `-update.1` indicates our increment number (starting at 1 and incrementing with each new release)

Each new release cumulatively includes all changes from previous releases on top of the current upstream state, ensuring that you always get the complete set of enhancements when using a specific tag.

### Dependency Management

**Important**: When using this fork in your projects, always reference **release tags** rather than commit hashes. This ensures proper dependency resolution since branches like `main-vxcontrol` undergo rebasing and their commit hashes change over time.

```
go get github.com/vxcontrol/langchaingo@v0.1.13-update.1
```

### Branch Visualization

```
  main              A---B---C---D---E---F   (synced with upstream)
                     \
  main-pull-requests   \---G---H---I        (rebased on main, PRs from upstream)
                          \
  main-vxcontrol           \---J---K---L    (default branch, rebased on main-pull-requests)
                                \
  release/vM.M.P-update.N        M          (tagged stable release)
```

### For Contributors

If you want to contribute to this fork, please create Pull Requests based on the current state of the `main-vxcontrol` branch. Even though commit hashes in this branch may change due to rebasing, your contributions will be included in the next release when enough changes have accumulated.

When creating a PR, please ensure your changes are well-tested and include appropriate documentation. Once merged, your contributions will be included in the next stable release with fixed commit hashes.

## Acknowledgements

Special thanks to [Travis Cline](https://github.com/tmc) (@tmc) and all [contributors](https://github.com/vxcontrol/langchaingo/graphs/contributors) who have made this project possible.

## Original resources

- Documentation: [pkg.go.dev/github.com/tmc/langchaingo](https://pkg.go.dev/github.com/tmc/langchaingo)
- Discord: [Join the official LangChain Go community](https://discord.gg/8bHGKzHBkM)

## 🤔 What is this?

This is the Go language implementation of [LangChain](https://github.com/langchain-ai/langchain).

## 📖 Documentation

- [Documentation Site](https://vxcontrol.github.io/langchaingo/docs/)
- [API Reference](https://pkg.go.dev/github.com/vxcontrol/langchaingo)


## 🎉 Examples

See [./examples](./examples) for example usage.

```go
package main

import (
  "context"
  "fmt"
  "log"

  "github.com/vxcontrol/langchaingo/llms"
  "github.com/vxcontrol/langchaingo/llms/openai"
)

func main() {
  ctx := context.Background()
  llm, err := openai.New()
  if err != nil {
    log.Fatal(err)
  }
  prompt := "What would be a good company name for a company that makes colorful socks?"
  completion, err := llms.GenerateFromSinglePrompt(ctx, llm, prompt)
  if err != nil {
    log.Fatal(err)
  }
  fmt.Println(completion)
}
```

```shell
$ go run .
Socktastic
```

# Resources

Join the Discord server for support and discussions: [Join Discord](https://discord.gg/8bHGKzHBkM)

Here are some links to blog posts and articles on using Langchain Go:

- [Using Gemini models in Go with LangChainGo](https://eli.thegreenplace.net/2024/using-gemini-models-in-go-with-langchaingo/) - Jan 2024
- [Using Ollama with LangChainGo](https://eli.thegreenplace.net/2023/using-ollama-with-langchaingo/) - Nov 2023
- [Creating a simple ChatGPT clone with Go](https://sausheong.com/creating-a-simple-chatgpt-clone-with-go-c40b4bec9267?sk=53a2bcf4ce3b0cfae1a4c26897c0deb0) - Aug 2023
- [Creating a ChatGPT Clone that Runs on Your Laptop with Go](https://sausheong.com/creating-a-chatgpt-clone-that-runs-on-your-laptop-with-go-bf9d41f1cf88?sk=05dc67b60fdac6effb1aca84dd2d654e) - Aug 2023


# Contributors

There is a momentum for moving the development of langchaingo to a more community effort, if you are interested in being a maintainer or you are a contributor please join our [Discord](https://discord.gg/8bHGKzHBkM) and let us know.

<a href="https://github.com/vxcontrol/langchaingo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tmc/langchaingo" />
</a>
