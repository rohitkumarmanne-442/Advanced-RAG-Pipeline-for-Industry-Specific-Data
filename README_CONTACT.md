# Contact Form Integration

A reusable inline contact form has been added under `src/contact/`.

## Wiring it into `app.py`

Add these two lines immediately **before** the existing footer markdown
block at the bottom of `app.py` (the `<div class="footer">…</div>` HTML):

```python
from src.contact.app_integration import mount_contact_section
mount_contact_section()
```

That's it. The form:

- Uses `st.form` so it only submits on the **Send** button click.
- Validates that Name / Email / Message are all non-empty
  (`Please fill in all fields.`).
- Validates the email format with a regex
  (`Please enter a valid email address.`).
- On success, shows a green `st.success` banner, appends the payload as
  one JSON object per line to `contacts.log` in the project root, and
  clears the form fields on the next rerun.
- Reuses the existing `.section-header` / `.section-icon` classes and a
  new `.contact-card` style that matches the app's palette and
  border-radius (`16px`, `#e2e8f0` border, indigo focus ring).

## Log format

Each line in `contacts.log` is a JSON object:

```json
{"timestamp": "2025-01-01T12:00:00+00:00", "name": "Jane Doe", "email": "jane@example.com", "message": "Hello"}
```

Add `contacts.log` to `.gitignore` if you don't want submissions committed.

## Tests

```bash
pytest tests/test_contact_form.py -v
```
