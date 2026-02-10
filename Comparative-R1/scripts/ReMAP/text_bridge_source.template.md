# ISIC Text Bridge Source (Template)

Fill this file with auditable domain text before running `build_text_rule_samples.py`.

## Label list
- Actinic keratosis
- Basal cell carcinoma
- Benign keratosis
- Dermatofibroma
- Melanocytic nevus
- Melanoma
- Squamous cell carcinoma
- Vascular lesion

## Diagnostic clues
- Add concise, source-grounded statements per label.
- Prefer statements that can be converted to yes/no checks.
- Keep wording stable and avoid ambiguous long narratives.

## Exclusion rules
- Add differential diagnosis exclusions.
- Use explicit words like "less likely when ..." / "exclude if ...".

## Priority rules
- Add rules that distinguish confusing pairs.
- Keep each rule short and testable.
