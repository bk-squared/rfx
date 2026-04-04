# 2026-04-04 PyPI Publish Complete

## Summary

`rfx` is now published on PyPI as:

- package name: `rfx-fdtd`
- version: `1.0.0`

The Python import remains:

```python
import rfx
```

## Follow-up

Repo-local and public docs should prefer:

```bash
pip install rfx-fdtd
```

and avoid stale `pip install rfx` wording, since `rfx` on PyPI is a different,
older package namespace owned by another project.
