# End-to-end evaluation fixtures

Suites are JSON objects with a unique `name`, an optional default `document`,
and a `tests` array. Each test defines `id`, `question`,
`fresh_conversation`, and an `expect` object. A deliberate follow-up can set
`fresh_conversation` to `false` and declare `setup_case_ids` for isolated runs.

Supported deterministic expectations include:

- `document` and `target` (`type`, `number`, `page`, `status`)
- `required_terms`
- `required_facts`, `required_concepts`, and `semantic_requirements`, using
  `any_of` or `all_of` regular-expression patterns
- `numeric_requirements`, with `value`, `tolerance`, optional `operator`, and
  optional `context_any`
- `forbidden_terms` and `forbidden_claims`, optionally scoped with
  `scope_after` / `scope_before`
- `expected_domains` and `expected_provenance`
- `debug.allowed_code_paths`, `debug.max_missing_slots`, and
  `debug.consistency_errors`

Fixture expectations are contracts. Production code never imports or rewrites
them.
