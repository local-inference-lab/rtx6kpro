# Support and Upstream Routing

## Reproduction record

Include:

- community wiki commit, current model runbook, and relevant troubleshooting/topology pages;
- exact custom/derivative image digest or package ID and archive SHA-256;
- exact recommended image and exact immediate base when different;
- applied patch/file identities and engine/component commits;
- model repository and immutable revision;
- hardware, topology, power, clocks, driver, CUDA, PyTorch, NCCL, and cache state;
- full launch command and relevant environment;
- minimal request, script, or sequence;
- expected and actual behavior;
- logs, traces, outputs, and hashes;
- first known working and failing identities when known.

## Custom image support

Reports concerning a custom image stay in its dedicated support thread until one upstream escalation condition is met.

The custom image author owns first-line diagnosis. Reproduce the failure on both:

1. the exact custom image; and
2. the current recommended image under an equivalent model/configuration.

The immediate base may also be tested when it differs from the recommended image, but passing or failing on another custom derivative does not replace the recommended-image control.

## Upstream escalation

Escalate only when either condition is satisfied:

1. the problem reproduces on the current recommended image under an equivalent configuration; or
2. a minimal source-level reproducer identifies the responsible source change independently of the custom image composition.

When neither condition is satisfied, keep ownership with the custom image and mark upstream attribution `UNKNOWN — needs verification`.

## Routing matrix

| Custom image | Recommended image | Minimal source reproducer | Route |
|---|---|---|---|
| Fails | Passes | Absent | Custom image support thread and author |
| Fails | Fails | Optional | Narrow upstream owner with both reproductions |
| Fails | Not tested | Identifies source | Narrow upstream owner plus custom thread |
| Fails | Not tested | Absent | Continue custom-image investigation |
| Passes | Fails | Optional | Recommended image/source owner |
| Configurations differ | Any | Any | Normalize configurations before attribution |

A vLLM failure is not automatically a B12X failure. A benchmark-method failure belongs with the benchmark tool. A Docker composition failure belongs with the image author. SGLang results are not evidence about vLLM and vice versa.

## Non-GitHub delivery

A complete reproducer package can be posted directly in Discord or another file-sharing route. GitHub is used only when the user selects it or upstream ownership is established.
