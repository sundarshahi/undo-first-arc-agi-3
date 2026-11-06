# arXiv paste sheet

Title:
Undo-First Exploration: Reversibility-Aware Frontier Search for ARC-AGI-3

Authors:
Sundar Shahi Thakuri

Abstract:
ARC-AGI-3 scores an agent by how many actions it needs to beat each level of an unseen game, squared against a human baseline. Every probing action costs, and an action that cannot be taken back can cost a reset and everything since the level started. We present undo-first exploration, a training-free, LLM-free agent that makes reversibility the organising signal for exploration. It builds a transition graph of each level, learns online which action types can be undone and which cells are only a UI counter, probes likely-undoable actions first, and picks the next frontier by travel cost plus the expected cost of getting back. On the 25 public games, with 10,000 actions per game and 3 seeds, it completes 22 of 183 levels (median), against 19 for the same explorer without the reversibility terms and 7 for random play, and finds that 29% of tested transitions had no known way back. Code and an offline notebook that reproduces every number are public.

Comments:
4 pages, 3 figures. ARC Prize 2026 paper track entry. Code and notebook: https://github.com/sundarshahi/undo-first-arc-agi-3

Primary category: cs.AI (Artificial Intelligence)
Cross-list: cs.LG (Machine Learning)
License: CC BY 4.0
Upload: arxiv-bundle.zip (paper.tex, numbers.tex, figures/*.pdf)
