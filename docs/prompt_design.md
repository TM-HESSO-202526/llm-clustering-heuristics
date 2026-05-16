# Prompt design

The notebook-equivalent runner uses one fixed system prompt and one dynamic user prompt per LLM call.

The prompt changes according to:

- active objective: `sse`, `pmedian`, or `radius`;
- parent-selection strategy: `1+1` or `1,1`;
- whether a full-valid parent exists;
- whether invalid-parent redesign mode is active.

The cleaned current version keeps invalid-parent code exposed when `hide_invalid_parent_code: false`, but removes the older objective-specific redesign fallback paragraphs. For Run C, the prompt keeps the sentence:

```text
If you maintain nearest-distance arrays, use Euclidean distances/radii that support the active radius objective.
```

All prompts are saved under each run folder in `prompts/`.
