# Frontend end-to-end test checklist (Person 6)

Record the date, tester, and result (pass/fail + note) for each row. A teammate
must run this on a clean environment before Day 39.

| # | Scenario | Steps | Expected | Result |
|---|----------|-------|----------|--------|
| 1 | Launch | `streamlit run app/streamlit_app.py` | Page loads, disclaimer visible, empty-state message shown | |
| 2 | Demo mode | Turn on Demo mode, upload a WAV, click Predict | Amber "Mock output" tag and dashed transcript; never looks like a real result | |
| 3 | Real prediction | Demo mode off, backend running, upload approved WAV | Transcript, model version tag, disclaimer | |
| 4 | Connection test | Sidebar: Test connection (backend up, then down) | Green when up, readable red message when down | |
| 5 | Unsupported file | Upload a non-WAV file (use the uploader's "all files" fallback / rename) | Clear message, no crash | |
| 6 | Empty / corrupt WAV | Upload a 0-byte or damaged .wav | Clear message, no crash | |
| 7 | Too large / too long | Upload a file over the size or duration limit | Message states the limit | |
| 8 | Backend down | Stop the backend, click Predict | "Can't reach the analysis service", no stack trace | |
| 9 | Backend error | Backend returns 422/500 | Backend message or status shown, no raw body | |
| 10 | Slow backend | Backend delays beyond timeout | "Took too long" message | |
| 11 | Confidence null | Response has `confidence: null` | No score line is shown | |
| 12 | Confidence present | Response has a score and type | Score and type shown with caveat | |
| 13 | XAI unavailable | `explanation.available=false` | Info box with the reason, no image | |
| 14 | XAI available | `available=true` with `visualization_url` | Method, summary, image, limitations expander | |
| 15 | Warnings | Response has warnings | Each is shown as a warning | |
| 16 | Rerun stability | Interact with sidebar after a result | Result stays, no duplicate API call | |
| 17 | HTML safety | Transcript contains `<b>x</b>` | Shown as text, not rendered | |
| 18 | Keyboard | Tab through all controls | Visible focus ring, all reachable | |
| 19 | Mobile width | Resize to ~380px | Layout readable, no sideways scroll | |
| 20 | Clean setup | Fresh clone + README commands | App runs without extra steps | |

## Bug list
| ID | Found in | Description | Severity | Owner | Status |
|----|----------|-------------|----------|-------|--------|
