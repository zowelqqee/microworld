# Narrative corpus drop zone

Put UTF-8 plain-text prose works here, one work per `.txt` file. Keep dialogue
and paragraph boundaries; remove page numbers, OCR noise, and markup when
possible. The ingest layer learns exact phrase edges plus universal action/object
frames shared by multiple subjects.

Build the merged narrative artifact with:

```bash
python3 cli.py ingest-narrative --source corpus
```

Use texts you are allowed to process. The repository does not download or
bundle third-party copyrighted books.
