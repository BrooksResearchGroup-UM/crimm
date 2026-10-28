## What and why

<!-- One concern per pull request. Say why the change is made, not only what it does. -->

Closes #

## Checklist

- [ ] Tests added or updated (a bug fix has a test that fails without it)
- [ ] `pytest -q` passes offline
- [ ] `import crimm` still needs only core dependencies
- [ ] Docstrings state units and whether arguments are modified in place
- [ ] Changelog entry for any user-visible change
- [ ] Stan's files touched (`CODEOWNERS`): wait for his approval before merging; otherwise merge after green CI and label `review-after-merge` (CONTRIBUTING, "Review and merge")
