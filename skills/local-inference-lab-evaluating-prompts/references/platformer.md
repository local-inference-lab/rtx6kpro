# Single-Page Platformer One-Shot Prompt

## Canonical prompt

```text
Build me an incredible, high fidelity clone of the original SMB, Level 1-1 and 1-2, as a single page web app. It needs to be faithful to the original, and feature music, good animations, sprites, correct physics, etc. etc, but not to the extent that you try to use original assets.
```

Keep the prompt exact for comparable runs. The result must use original replacement assets rather than copied game art, audio, maps, or other copyrighted source assets.

## Protocol

Run as one prompt and one turn unless the user explicitly asks for an agentic or iterative version. Record browser/tool access, package installation, token limit, reasoning effort, and whether the model could preview the app.

## Evaluation rubric

### Build and launch

- The single-page application installs/builds and starts with documented commands.
- No required source, asset, or configuration is missing.
- The browser console remains free of fatal errors during play.

### Game systems

- Movement, acceleration, jumping, gravity, collision, camera, death, restart, and level transitions are coherent.
- Enemies, blocks, collectibles, hazards, pipes, and goals behave consistently.
- Level 1-1 and 1-2 are recognizably distinct and complete enough to play through.

### Fidelity without copied assets

- Layout, pacing, physics, and interaction evoke the requested reference.
- Sprites, audio, and visual design are newly created or procedurally generated.
- The output does not download or embed original proprietary assets.

### Presentation

- Animation, parallax/camera behavior, music, sound effects, HUD, and responsive layout work together.
- Controls and start/restart states are clear.

### Evidence

Preserve generated files, dependency lock where produced, start command, console output, and a playthrough recording. List any manual fixes outside the one-shot result.
