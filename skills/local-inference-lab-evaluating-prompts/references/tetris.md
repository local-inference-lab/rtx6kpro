# Tetris One-Shot Prompt

## Canonical prompt

```text
Make me an amazing mind blowing tetris clone with cool music and awesome visuals
```

Keep the wording exact for comparable runs. Store it as `prompt.txt` and hash it.

## Protocol

Use a single prompt in one turn without an agentic repair loop as the standard comparison protocol.

Record whether the model returned one file or several files and whether the harness automatically executed or previewed the result.

## Evaluation rubric

Record observations rather than inventing a single objective score.

### Functional

- The project builds or opens without errors.
- Tetrominoes spawn, fall, move, rotate, lock, and clear completed lines.
- Collision, boundaries, scoring, game-over, and restart work.
- Keyboard or on-screen controls are discoverable.

### Completeness

- The output is runnable from the supplied files.
- No essential source is omitted or replaced by prose.
- Required dependencies and start command are clear.

### Presentation

- Visuals are coherent and responsive.
- Music/audio is present or the limitation is explicit.
- Animation communicates movement, line clears, and game state.
- Polish does not hide broken core mechanics.

### Evidence

Capture the exact generated files, build/start command, console errors, and a short screen recording or screenshots. Record any manual changes separately; they are not part of the one-shot result.
