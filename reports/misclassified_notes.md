# Notes on the 15 sampled test-set errors

The examples, labels and probabilities come from `reports/test_results.md`
(produced by `python -m src.evaluate_test`). The notes below are a human
reading of each ticket, not script output: they're guesses about *why* the
model missed, and some say the label itself looks questionable.

| # | True → Predicted | Likely reason |
|---|---|---|
| 1 | Miscellaneous → Access | "creation users" reads like an account-creation (Access) request. The label may be the odd one out; the text doesn't explain why it's Miscellaneous. |
| 2 | Hardware → HR Support | "new headset" plus "new … march pm" resembles the HR "new starter" template. The one clear hardware word, *headset*, didn't outweigh the template words. |
| 3 | HR Support → Hardware | A website redirect/deployment request (mentions "careers" pages). It's not obviously HR or Hardware. Possibly labelled HR because it concerns the careers site: the topic, not the type of work. |
| 4 | Hardware → Miscellaneous | A reporting-dashboard data problem. Nothing about physical hardware; Miscellaneous seems at least as reasonable. Probable label noise. |
| 5 | HR Support → Hardware | "configure jobs … job views": very little content once stopwords are gone, so the model is guessing (confidence 0.66). |
| 6 | Access → Miscellaneous | A database sync incident. No access-request words ("grant", "rights", "permission"), so the text gives little hint of the Access label. |
| 7 | Access → HR Support | Delegating expense/corporate-card approval while someone is on holiday. "Holiday", "expenses" and "corporate card" are HR-flavoured words, though the request is for permissions (Access). Close call (0.57 vs 0.38). |
| 8 | Hardware → HR Support | A suspicious / phishing email report. Security doesn't fit any of the 8 categories well, so neither label really fits. |
| 9 | HR Support → Hardware | The text starts with "access" and describes something not being accessible at a location. It doesn't look like HR. Probable label noise. |
| 10 | Hardware → Miscellaneous | "amend team membership": a group-membership change, closer to Access than to either label. Low confidence (0.46). |
| 11 | Hardware → Internal Project | "access to project codes … cube report" strongly matches Internal Project vocabulary. The model is very confident (0.93) and the label looks wrong. |
| 12 | Access → HR Support | A request to borrow an Android tablet for testing. Arguably Hardware; the model said HR, the label says Access. Shows how overlapping the categories are. |
| 13 | Access → HR Support | Asks for repository rights and permission to add members, which is clearly Access in wording. Near miss (0.51 vs 0.45). Generic polite words ("please help", "thank") are common in HR tickets. |
| 14 | Miscellaneous → Hardware | "cannot execute … contract": too vague to route without more context. |
| 15 | HR Support → Access | "new starter access card" contains "access" *and* "new starter". The model went with Access (0.98). Both readings are defensible; this is a category-boundary problem, not a model bug. |

## Patterns

- **Questionable labels:** in roughly 5 of the 15 (#4, #8, #9, #11, and arguably #3), the label itself looks wrong or arbitrary to a human reader. Label noise sets a ceiling on any model's score on this dataset.
- **Overlapping categories:** "Access" vs "HR Support" (new-starter access cards, delegations) and "Hardware" vs "Miscellaneous" are the main boundaries. The confusion matrix shows the same: the most common errors are HR Support→Hardware (107) and Hardware→Miscellaneous (106).
- **Too little text:** several errors are short tickets that lost most of their meaning when stopwords were stripped upstream (#5, #14).
- **Hardware is the default sink:** when the model is unsure it leans toward Hardware, the largest class. 11% of Administrative-rights tickets go there.
