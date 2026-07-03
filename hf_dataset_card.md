# Arabic Medical OCR Corrections

---
license: mit
task_categories:
  - text-correction
language:
  - ar
size_categories:
  - n<1K
pretty_name: Arabic Medical OCR Corrections
---

## Dataset Description

Arabic medical OCR error-correction pairs collected from real handwritten pediatric orthopedic documents. Each row contains an incorrect OCR output alongside the human-verified correct text.

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `incorrect_ocr_output` | `string` | The raw OCR engine output (contains errors) |
| `correct_text` | `string` | Human-verified correct text |
| `category` | `string` | Error type: `spelling`, `missing_char`, `extra_char`, `word_split`, `word_merge`, `diacritic`, `other` |
| `form` | `string` | Text form: `printed`, `handwritten`, `mixed` |

## Languages

Primary: **Arabic** (ar) — with some English medical terminology mixed in.

## Uses

- Training post-OCR correction models
- Building spell-correction dictionaries for Arabic medical text
- Benchmarking OCR correction quality
- Fine-tuning LLM-based proofreaders (e.g., Jais, AraBERT)

## Limitations

- Domain-specific to pediatric orthopedic medical text
- Arabic dialectal variations may not be covered
- Small dataset (< 1K pairs) — suitable for fine-tuning, not training from scratch
- Mixed Arabic/English terms may have ambiguous corrections

## License

MIT