# Probe examples (hand-written tickets, not an evaluation)

| Ticket | Top 3 (confidence) | Words/phrases the model knows |
|---|---|---|
| Hi, my laptop will not connect to the docking station monitors since this morning. | Hardware 0.99, Administrative rights 0.01, Purchase 0.00 | hi, laptop, not, connect, to, the, docking, station, monitors, since, this, morning |
| I forgot my password and am locked out | Access 1.00, Administrative rights 0.00, Hardware 0.00 | forgot, password, and, am, locked, out |
| please add me to the shared drive folder | Storage 1.00, Miscellaneous 0.00, Administrative rights 0.00 | please, add, me, to, the, shared, drive, folder |
| New starter joining Monday, please set up onboarding. | HR Support 0.99, Internal Project 0.01, Administrative rights 0.00 | new, starter, joining, monday, please, set, up |
| Please grant me admin rights on my machine so I can install Visual Studio. | Access 0.84, Administrative rights 0.07, Storage 0.04 | please, me, admin, rights, on, machine, can, install, visual, studio |
| We need to order 5 new keyboards and a monitor for the new team, PO attached. | Hardware 0.87, HR Support 0.06, Purchase 0.05 | we, need, to, order, new, keyboards, and, monitor, for, the, new, team, po, attached |
| google app is frozen | Administrative rights 0.37, Hardware 0.37, HR Support 0.11 | app, is, frozen |
| my google app is frozen and wont open | Hardware 0.46, Administrative rights 0.29, HR Support 0.12 | app, is, frozen, and, wont, open |
| the google app is frozen on my laptop | Hardware 1.00, Administrative rights 0.00, Access 0.00 | the, app, is, frozen, on, laptop |
| Google Chrome keeps freezing and crashing | Hardware 0.55, Administrative rights 0.20, HR Support 0.12 | chrome, keeps, freezing, and, crashing |
| outlook is not opening | Administrative rights 1.00, Storage 0.00, Miscellaneous 0.00 | outlook, is, not, opening |
| zzz | Hardware 0.37, Miscellaneous 0.18, HR Support 0.17 | (none) |

## How the training data labels some words

| Word | Training tickets containing it | Top labels (% of those tickets) |
|---|---:|---|
| frozen | 8 | Hardware 88%, HR Support 12% |
| freezing | 15 | Hardware 87%, Administrative rights 13% |
| crashing | 16 | Hardware 75%, Administrative rights 19%, Access 6% |
| chrome | 202 | Hardware 58%, HR Support 20%, Access 8% |
| software | 967 | Hardware 38%, Access 17%, Administrative rights 16% |
| install | 688 | Hardware 43%, Administrative rights 23%, Access 15% |
| outlook | 460 | Administrative rights 76%, Storage 8%, Hardware 7% |
| google | 0 | - |

## What the model predicts when it knows no words at all

With an all-zero feature vector only the intercepts count, and the largest is **Hardware**: Hardware 1.45, Miscellaneous 0.71, HR Support 0.65, Access 0.18, Administrative rights -0.29, Storage -0.59, Internal Project -1.04, Purchase -1.07
