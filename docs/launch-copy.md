# StrAIght Shot — Launch Copy

## Short (Twitter)

We built the thing the papers said was possible.

StrAIght Shot reads what language models think but don't say — detecting jailbreaks, prompt injections, and hallucination onset in real-time by monitoring internal activations.

Three channels. One middleware. MIT licensed.

github.com/noirsys/straightshot

## Medium (Twitter thread teaser)

1/ The Jacobian lens paper dropped 4 days ago. "Reference only, not maintained."

We built it anyway.

Meet StrAIght Shot — real-time safety monitoring that reads LLM internal activations. Before the model outputs anything, we know if it's safe.

github.com/noirsys/straightshot

## Long (blog post / LinkedIn)

Title: We Built What the Papers Said Was Possible

Last week, Anthropic published a paper showing that language models have a "global workspace" — internal representations that reveal what the model is thinking about, even when it never says it aloud.

The paper called it a Jacobian lens. The reference code said "not maintained, not accepting contributions."

We spent the weekend building anyway.

StrAIght Shot is a real-time middleware layer that monitors LLM internal activations during generation. It uses three parallel detection channels:
- A linear probe trained to recognize unsafe hidden-state patterns
- The Jacobian lens to decompose internal activations into concept space
- Entropy dynamics to detect reasoning collapse before it happens

The result: per-token safety classification that catches jailbreaks, prompt injections, and hallucination onset — before the model writes the token.

It's live. It's streaming. It's on GitHub.

MIT license. Contributions welcome.

github.com/noirsys/straightshot

## Taglines (short form)

- "StrAIght Shot reads what language models think but don't say."
- "Three channels. One verdict. Every token."
- "Before the token leaves the model, StrAIght Shot knows if it's safe."
- "The first real-time LLM safety layer that looks through the skull."
