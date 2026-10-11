# Data profile (raw data)

Source file: `all_tickets_processed_improved_v3.csv`

## Shape
- Rows: 47,837
- Columns: ['Document', 'Topic_group']

## Missing or empty values
- Null text: 0
- Null or whitespace-only text: 0
- Null label: 0

## Class distribution
| Class | Count | % |
|---|---:|---:|
| Hardware | 13,617 | 28.47 |
| HR Support | 10,915 | 22.82 |
| Access | 7,125 | 14.89 |
| Miscellaneous | 7,060 | 14.76 |
| Storage | 2,777 | 5.81 |
| Purchase | 2,464 | 5.15 |
| Internal Project | 2,119 | 4.43 |
| Administrative rights | 1,760 | 3.68 |

- Number of classes: 8
- Largest / smallest class ratio: 7.7x
- Majority-class share (what always guessing 'Hardware' scores as accuracy): 28.47%

## Text length
| Quantile | Words | Characters |
|---|---:|---:|
| 0% | 2 | 7 |
| 5% | 10 | 60 |
| 25% | 17 | 110 |
| 50% | 26 | 175 |
| 75% | 46 | 304 |
| 95% | 136 | 926 |
| 99% | 284 | 1933 |
| 100% | 981 | 7015 |

- Tickets with 1-2 words: 14
- Tickets with 3-5 words: 273

### Median words per ticket by class
| Class | Median words |
|---|---:|
| Storage | 21 |
| Access | 22 |
| Internal Project | 24 |
| HR Support | 24 |
| Miscellaneous | 27 |
| Administrative rights | 30 |
| Purchase | 30 |
| Hardware | 32 |

## Exact duplicates (after collapsing whitespace)
- Extra copies of an identical text with the same label: 0
- Distinct texts that appear with MORE THAN ONE label: 0 (covering 0 rows)

## Near duplicates (TF-IDF cosine >= 0.9, among unique texts)
- Near-duplicate pairs: 62,072
- Unique texts involved in at least one pair: 7,209 of 47,837
- Pairs where both texts have the same label: 58,993 (95.0%)

- Near-duplicate clusters (2+ texts chained together): 1,638
- Largest 5 cluster sizes: [638, 236, 152, 77, 62]
  - cluster of 638: labels {'Purchase': 638}
  - cluster of 236: labels {'HR Support': 236}
  - cluster of 152: labels {'Hardware': 115, 'Storage': 18, 'Miscellaneous': 18, 'Access': 1}

Examples:

- sim 0.90: `new purchase po wednesday july pm purchase po dear purchased has please log allo` (Purchase) vs `new purchase po wednesday pm purchase po dear purchased speak please log allocat` (Purchase)
- sim 0.92: `new starter bucharest thursday october pm re va ne si va la wednesday october pm` (HR Support) vs `new starter bucharest thursday pm re va ne si va la wednesday pm hello va ne si ` (HR Support)
- sim 0.95: `leaver november center leaver form leave date november location center hello lea` (HR Support) vs `leaver form for leave date july location kale building wednesday july leaver for` (HR Support)
- sim 0.97: `leaver centre leaver form leave date location centre hello leaver form has fille` (HR Support) vs `leaver form for leave date october location tower leaver form leave date october` (HR Support)
- sim 0.98: `leaver account leaver form leave date location hello leaver form has filled by n` (HR Support) vs `leaver sonar march tower wednesday march leaver form leave date march location t` (HR Support)

## Possible label leakage
- Columns other than text and label: none
- Share of tickets whose text contains their own class name as a word:

  - Hardware: 1.4%
  - HR Support: 0.0%
  - Access: 27.7%
  - Miscellaneous: 0.0%
  - Storage: 9.3%
  - Purchase: 68.0%
  - Internal Project: 0.5%
  - Administrative rights: 0.0%

## Personal information scan
| Pattern | Tickets matching | Example matches |
|---|---:|---|
| email address | 0 | - |
| email-like 'at' token pair | 4 | 'ms at ms com', 'files at one com', 'approved at oracle com', 'error at oracle com' |
| phone-like digit run (7+ digits) | 0 | - |
| any digit | 0 | - |
| url | 0 | - |
| ip address | 0 | - |

Most common words right after a greeting or sign-off (where names usually appear):

please (21529), analyst (3688), engineer (3293), senior (3218), guys (2459), administrator (2316), kind (2309), sent (1669), purchased (1568), manager (1413), have (1358), could (1316), officer (1280), best (1193), head (1154), ext (1109), can (1098), thank (843), regards (786), leader (784), for (737), we (684), wednesday (675), tuesday (664), kindly (626), everyone (620), thursday (603), colleagues (591), tester (586), help (579), received (558), friday (541), developer (526), specialist (509), re (506), infrastructure (496), lead (450), dear (404), thanks (398), application (388)

Common first names found as words (checked 64 names): {'ion': 52, 'florin': 40}, in 91 tickets
