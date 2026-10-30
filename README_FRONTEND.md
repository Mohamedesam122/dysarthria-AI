# Dysarthria AI: Frontend (Streamlit)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-frontend.txt
```

## Run
Demo mode (no backend needed; results are labelled placeholders):
```bash
DYSARTHRIA_MOCK=1 streamlit run app/streamlit_app.py      # Windows PowerShell: $env:DYSARTHRIA_MOCK=1
```
Against the real backend:
```bash
export DYSARTHRIA_API_URL=http://localhost:8000
streamlit run app/streamlit_app.py
```
All settings are in `.env.example`.

## Tests
```bash
python -m unittest discover -s tests
```
Tests use mocks. They need no dataset, model or running backend.

## Contract to confirm with Person 5 (Day 5)
- `GET /health` returns 200.
- `POST /predict`, multipart field `file`, returns the JSON in the project doc, section 8.3.
- Errors return a non-2xx status with `{"message": "..."}` or `{"detail": "..."}`.
- `visualization_url` may be absolute or start with `/` (served by the backend).
If the path names differ, set `DYSARTHRIA_PREDICT_PATH` / `DYSARTHRIA_HEALTH_PATH`.

## Layout
- `app/streamlit_app.py`: UI and states (empty, loading, success, error, unsupported file).
- `app/api_client.py`: the only place that knows the API contract; validation, mock, error handling.
- `docs/frontend_e2e_checklist.md`: end-to-end test sheet and bug list.

Do not commit uploads, `.env`, datasets or checkpoints.
