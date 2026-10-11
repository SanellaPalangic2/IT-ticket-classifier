# Split report

Seed: 42. Clusters of 50+ tickets forced to train: 10 clusters, 1,441 rows. Remaining rows: StratifiedGroupKFold, 20 folds dealt at random (14 train / 3 val / 3 test).

| Split | Rows | % of total | Groups |
|---|---:|---:|---:|
| train | 33,920 | 70.9 | 29,587 |
| val | 6,958 | 14.5 | 6,336 |
| test | 6,959 | 14.5 | 6,338 |

## Class mix per split (% of that split)

| Class | All | Train | Val | Test |
|---|---:|---:|---:|---:|
| Hardware | 28.47 | 28.20 | 29.10 | 29.10 |
| HR Support | 22.82 | 22.91 | 22.59 | 22.59 |
| Access | 14.89 | 14.76 | 15.23 | 15.23 |
| Miscellaneous | 14.76 | 14.59 | 15.18 | 15.17 |
| Storage | 5.81 | 5.75 | 5.94 | 5.95 |
| Purchase | 5.15 | 5.79 | 3.59 | 3.59 |
| Internal Project | 4.43 | 4.37 | 4.57 | 4.57 |
| Administrative rights | 3.68 | 3.63 | 3.79 | 3.79 |

## Class counts per split

| Class | Train | Val | Test |
|---|---:|---:|---:|
| Hardware | 9,567 | 2,025 | 2,025 |
| HR Support | 7,771 | 1,572 | 1,572 |
| Access | 5,005 | 1,060 | 1,060 |
| Miscellaneous | 4,948 | 1,056 | 1,056 |
| Storage | 1,950 | 413 | 414 |
| Purchase | 1,964 | 250 | 250 |
| Internal Project | 1,483 | 318 | 318 |
| Administrative rights | 1,232 | 264 | 264 |

## Where the 5 biggest near-duplicate clusters went

- group 24: 638 tickets, label(s) ['Purchase'], split: ['train']
- group 48: 236 tickets, label(s) ['HR Support'], split: ['train']
- group 10918: 152 tickets, label(s) ['Access', 'Hardware', 'Miscellaneous', 'Storage'], split: ['train']
- group 85: 77 tickets, label(s) ['HR Support'], split: ['train']
- group 498: 62 tickets, label(s) ['HR Support', 'Miscellaneous'], split: ['train']

## Checks

- PASS: every row assigned to exactly one split
- PASS: no group_id shared between splits
- PASS: no identical text shared between splits