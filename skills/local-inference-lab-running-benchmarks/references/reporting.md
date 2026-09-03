# Benchmark Reporting

A report is an evidence index, not a screenshot collection.

## Required identity block

Record:

- community wiki repository and full commit;
- current model runbook path and commit-pinned URL;
- related wiki methodology pages and their relationship to the new run;
- benchmark repository and full commit;
- candidate and control image digests or source commits;
- engine and component commits;
- model repository and immutable revision;
- quantization/activation/KV formats;
- complete hardware, topology, power, clocks, and runtime versions;
- launch commands, environment differences, graph/cache/scheduler configuration;
- exact benchmark commands and raw-result hashes.

## Separate measurement families

Report these independently:

1. prefill;
2. target-only sustained decode;
3. finite burst or end-to-end request throughput;
4. speculative decode;
5. completion-token/reasoning profiles;
6. dataset accuracy;
7. qualitative prompt observations.

Do not merge these into one headline speed number.

## Conditions, measurement, result, conclusion

Every claim must state:

```text
Conditions: exact candidate/control configuration
Measurement: command, repetitions, aggregation, exclusions
Result: values with units and raw evidence
Conclusion: narrow statement supported by the result
```

## Percentiles and repetitions

For p50/p95/p99 or outlier claims:

- identify the sampled quantity;
- state sample count and run count;
- publish each independent run;
- state pooled versus per-run aggregation;
- do not average percentiles without saying so;
- report failures, cancellations, truncations, and excluded samples.

## Comparable performance claims

A performance claim requires:

- digest-pinned or commit-pinned candidate and control;
- identical benchmark commit and workload;
- exact candidate and control commands;
- same hardware and serving conditions;
- input lengths, output length or duration, concurrency, and run count;
- the complete list of changed variables;
- raw result URLs/files and hashes.

When these conditions are not met, label the result exploratory or a system comparison.

## Missing information

Use exactly:

- `UNKNOWN — needs verification` for an unresolved fact;
- `Not tested` for an unrun test/configuration;
- `N/A` for a genuinely inapplicable field.

Do not remove template sections.

## Concise Discord summary

The short post should include only:

- decision tested;
- exact candidate and control identities;
- selected suite and material configuration;
- narrow result;
- important limitation;
- link or attached package containing commands and raw evidence.

Keep long commands and tables in the attached report.
