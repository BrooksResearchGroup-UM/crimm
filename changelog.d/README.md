# Changelog fragments

Every user-visible change adds one small file here instead of editing `CHANGELOG.md`, so
pull requests opened at the same time do not conflict (#180). At release, towncrier writes
the fragments into `CHANGELOG.md` and deletes them.

## Naming

`<issue>.<type>.md`, for example `43.fixed.md`. `<type>` is one of the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) sections:

| Type | Section |
| --- | --- |
| `added` | Added |
| `changed` | Changed |
| `deprecated` | Deprecated |
| `removed` | Removed |
| `fixed` | Fixed |
| `security` | Security |

- Two entries for one issue in the same section: `43.fixed.md` and `43.fixed.1.md`.
- An entry without an issue: `+<short-slug>.<type>.md`, for example `+test-extra.added.md`.
- `tests/test_changelog_fragments.py` rejects any other name.

## Content

One entry in Markdown, written for users: what changed and what they should do, not how the
code changed. No leading `- `; towncrier adds the bullet and the issue link.

## Release

```bash
pipx run towncrier build --draft --version 2026.3   # preview
pipx run towncrier build --version 2026.3           # write CHANGELOG.md, delete fragments
```

(`uvx towncrier` works too.) towncrier is not a dependency of crimm.
