# Output Evidence Index

Full simulated transcripts are intentionally stored outside Git under `/private/tmp/pfe-phase100-simulated-review`, `/private/tmp/pfe-phase101-simulated-review`, `/private/tmp/pfe-phase102-simulated-review`, and `/private/tmp/pfe-phase103-simulated-review`.

Repository evidence contains per-turn output hashes, termination reasons, scores, and aggregate metrics without private transcript text.

- Phase100: 24 final calls, complete native termination 1.0, provenance 1.0 with guided runtime target.
- Phase101: base format 0.5 versus SFT 0.0.
- Phase102: DPO metrics matched base and did not improve provenance.
- Phase103: base/DPO acceptance both 0.40; paired result 0 wins, 19 ties, 1 loss.
