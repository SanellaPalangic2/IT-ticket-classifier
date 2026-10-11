# Cleaning report

| Step | Rows after | Effect |
|---|---:|---|
| Raw rows | 47,837 |  |
| 1. Normalise whitespace | 47,837 | 174 texts changed |
| 2. Drop empty text | 47,837 | 0 rows dropped |
| 3. Drop exact duplicates | 47,837 | 0 rows dropped |
| 4. Mask first names | 47,837 | 91 texts changed |
| 5. Merge rare classes | 47,837 | 0 labels changed |
| 6. Assign near-duplicate groups | 47,837 | 42,261 groups; 7,215 rows are in 1,639 multi-ticket clusters |

## Class distribution after cleaning
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

Saved 47,837 rows to `data\processed\tickets_clean.csv`