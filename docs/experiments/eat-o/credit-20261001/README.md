# Failed exact credit launch

All eleven attempts failed at command-line parsing with `unknown option --raw-cache`.
No training or quality evaluation occurred. The runner copied the prior executable
before the new release build had completed. The protocol and failed events remain
unaltered; this directory contains no passing receipts or model-quality results.

The executable and source snapshot therefore did not describe the same build.
Their frozen hashes are evidence of the launch error, not implementation validity.
The corrected campaign is [credit-20261002](../credit-20261002/results.md), which
adds an executable preflight and retains the same scientific comparison.
