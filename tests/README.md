# Tests

Test folder ownership follows `PROMPTS.md`. Add analytical tests before randomized tests. Test expected values must be derived independently from implementation helpers.

Useful first command after T0:

```powershell
.\.venv\Scripts\python.exe scripts alidate_config.py
.\.venv\Scripts\python.exe -m pytest -q
```
