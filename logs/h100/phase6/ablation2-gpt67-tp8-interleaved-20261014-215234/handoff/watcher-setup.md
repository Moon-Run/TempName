# Five-group suite startup watcher

Session: `ablation2-gpt67-tp8-interleaved-20261014-215234`. One suite only: interleaved, native_taco/G1-G4, two rounds, five paired blocks per round, 50 formal windows.

The user authorized waiting after the previous startup resource rejection. Busy GPUs are checked every 600 seconds; three idle confirmations are spaced 10 seconds apart. A heartbeat is written every 30 seconds. No other task is stopped. All GPU polling stops when the suite is launched; the strict entry performs its final availability gate. Any failure stops the suite without an automatic retry. The earlier failed session is preserved.
